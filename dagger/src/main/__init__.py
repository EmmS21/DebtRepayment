from dagger import dag, function, object_type, Secret
import logging
import json
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re
import redis
from langchain.globals import set_debug
import requests
from typing import List
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import base64
import math


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@object_type
class DebtRepayment:
    @function
    async def fetch_data(self, apiKey: Secret, sheet: Secret, sheet_two: Secret, open_key: Secret, name: str) -> str:
        """
        Fetch data from two Google Sheets and run an agent to analyze the debts and investments.
        
        Args:
            apiKey (Secret): API key for accessing the Google Sheets API.
            sheet (Secret): Secret for the first Google Sheet.
            sheet_two (Secret): Secret for the second Google Sheet.
            open_key (Secret): API key for OpenAI.
            name (str): Name of the sheet to fetch data from.
        
        Returns:
            str: Cleaned decision output from the agent.
        """        
        fetch_balance = await dag.fetch_spreadsheet_data().fetch_data(apiKey, sheet, 'Edited')
        logger.info(f"balance_resp: {fetch_balance}")
        fetch_balance = self.convert_amounts(fetch_balance)
        fetch_debt = await dag.fetch_spreadsheet_data().fetch_data(apiKey, sheet_two, name)
        fetch_debt = self.restructure_data(fetch_debt)
        decision = await self.run_agent(open_key, fetch_debt)        
        output = decision.get("decision", {}).get("output", "")

        visualizations = decision.get("visualizations", {})
        debt_visualization = visualizations.get('debt_visualization', '')
        stock_visualization = visualizations.get('stock_visualization', '')

        # Save visualizations as HTML files
        with open('debt_visualization.html', 'w') as f:
            f.write(debt_visualization)

        with open('stock_visualization.html', 'w') as f:
            f.write(stock_visualization)

        # Print paths to visualization files for terminal testing
        print("Visualizations saved as 'debt_visualization.html' and 'stock_visualization.html'")

        clean_decision = re.sub(r'\\n', ' ', output)
        return clean_decision
    
    def restructure_data(self, data_str) -> str:
        """
        Restructure data fetched from Google Sheets for further processing.

        Args:
            data_str (str): JSON string containing raw data.

        Returns:
            str: JSON string of restructured data.
        """
        data = json.loads(data_str)
        categories = {"Balance": {}, "Min. Payment": {}, "Interest Rate": {}}
        for entry in data:
            category = entry.get("")
            if category in categories:
                for key, value in entry.items():
                    if key: 
                        if value:
                            try:
                                categories[category][key] = float(value)
                            except ValueError:
                                categories[category][key] = value
                        else:
                            categories[category][key] = value
        keys = list(categories["Balance"].keys())
        result = [[""] + keys]
        for category, values in categories.items():
            row = [category]
            for key in keys:
                row.append(values.get(key, ""))
            result.append(row)
        result_json = json.dumps(result)
        return result_json
    
    def process_balance_data(self, data: str) -> str:
        """
        Process balance data to extract specific columns and return as JSON string.

        Args:
            data (str): JSON string containing balance data.

        Returns:
            str: JSON string of processed balance data.
        """
        data_json = json.loads(data)
        if not data_json:
            return json.dumps([])
        required_columns = ['Interest Checking - Fixed (xxxx5222)', 'Savings Account (xxxx8919)']        
        processed_data = []
        for row in data_json:
            filtered_row = {col: row[col] for col in required_columns if col in row}
            if filtered_row:
                processed_data.append(filtered_row)
        return json.dumps(processed_data)

    def clean_amount(self, amount: str) -> float:
        """
        Remove dollar sign and commas from amount and convert to float.

        Args:
            amount (str): Amount string to be cleaned.

        Returns:
            float: Cleaned amount as float.
        """
        clean_amount = amount.replace('$', '').replace(',', '')
        return float(clean_amount)
    
    def convert_amounts(self, data: str) -> str:
        """
        Convert amounts in the fetched balance data to floats and return as JSON string.

        Args:
            data (str): JSON string containing balance data.

        Returns:
            str: JSON string of balance data with amounts converted to floats.
        """
        data_json = json.loads(data)
        if not data_json:
            return json.dumps([])
        for row in data_json:
            if 'Amount' in row:
                row['Amount'] = self.clean_amount(row['Amount'])
        return json.dumps(data_json)
    
    async def run_agent(self, open_key, fetch_debt) -> str:
        """
        Run an agent to analyze debts and investments and return a decision.

        Args:
            open_key (Secret): API key for OpenAI.
            fetch_debt (str): JSON string of fetched debt data.

        Returns:
            str: JSON string containing the decision from the agent.
        """
        def upload_image_to_host(image_bytes):
            """
            Upload image to a hosting service and return the URL.

            Args:
                image_bytes (bytes): Image bytes to be uploaded.

            Returns:
                str: URL of the uploaded image.
            """
            api_endpoint = "https://api.imgbb.com/1/upload"
            api_key = "9bb6bd78cb6c03a2dfddce20b69cc45c"
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            payload = {
                "key": api_key,
                "image": encoded_image,
            }
            response = requests.post(api_endpoint, data=payload)
            if response.status_code == 200:
                json_response = response.json()
                if json_response["success"]:
                    return json_response["data"]["url"]
                else:
                    raise Exception(f"Failed to upload image: {json_response['error']['message']}")
            else:
                raise Exception(f"Failed to upload image: {response.text}")
            
        @tool  
        async def fetch_stocks(sectors_of_interest:str) -> str:
            """
            Fetch stock data for given sectors.

            Args:
                sectors_of_interest (str): Comma-separated list of sectors.

            Returns:
                str: JSON string of fetched stock data.
            """
            stocks_data = await dag.get_stocks().stocks(sectors_of_interest,period=5, top=3)
            return json.dumps(stocks_data)
            
        def calculate_sharpe_ratio(monthly_returns, risk_free_rate_annual):
            """
            Calculate the Sharpe ratio for given monthly returns and risk-free rate.

            Args:
                monthly_returns (list): List of monthly returns.
                risk_free_rate_annual (float): Annual risk-free rate.

            Returns:
                tuple: Sharpe ratio, average monthly return, and standard deviation of monthly returns.
            """
            risk_free_rate_monthly = (1 + risk_free_rate_annual) ** (1/12) - 1
            average_monthly_return = sum(monthly_returns) / len(monthly_returns)
            mean_difference_squared = [(x - average_monthly_return) ** 2 for x in monthly_returns]
            variance = sum(mean_difference_squared) / len(monthly_returns)
            std_monthly_return = math.sqrt(variance)
            excess_return = average_monthly_return - risk_free_rate_monthly
            sharpe_ratio = excess_return / std_monthly_return if std_monthly_return != 0 else 0
            return sharpe_ratio, average_monthly_return, std_monthly_return
            
        def opportunity_cost_comparison(debts, stock_data, risk_free_rate_annual):
            """
            Compare opportunity costs between debts and stock investments.

            Args:
                debts (list): List of debt dictionaries.
                stock_data (list): List of stock data dictionaries.
                risk_free_rate_annual (float): Annual risk-free rate.

            Returns:
                tuple: Future value of debts, future value of stocks, and comparison results.
            """
            future_value_debts = {}
            future_value_stocks = {}
            comparisons = []

            for debt in debts:
                debt_name = debt['debt_name']
                interest_rate = debt['interest_rate'] / 100 / 12
                future_value_debt = 0
                normalized_balance = 1000
                for _ in range(12):
                    interest = normalized_balance * interest_rate
                    future_value_debt += round(interest, 2)
                future_value_debts[debt_name] = future_value_debt

            for stock in stock_data:
                stock_name = stock['symbol']
                annual_return = float(stock['average_return'])
                monthly_returns = stock['monthly_return']
                #too many values error
                sharpe_ratio, average_monthly_return = calculate_sharpe_ratio(monthly_returns, risk_free_rate_annual)

                monthly_return = (1 + annual_return) ** (1/12) - 1
                future_value_stock = 0
                normalized_investment = 1000
                for monthly_return_value in monthly_returns:
                    growth = normalized_investment * (1 + monthly_return_value)
                    future_value_stock += round(growth - normalized_investment, 2)  # Subtracting normalized_investment to get the growth
                    normalized_investment = growth  # Compounding for the next month
                future_value_stocks[stock_name] = future_value_stock

                for debt_name, debt_value in future_value_debts.items():
                    debt_interest_saving = debt_value
                    stock_return = future_value_stock
                    comparisons.append({
                        'debt_name': debt_name,
                        'stock_name': stock_name,
                        'debt_interest_saving': debt_interest_saving,
                        'stock_return': stock_return,
                        'sharpe_ratio': sharpe_ratio,
                        'average_monthly_return': average_monthly_return,
                        'monthly_return': monthly_return
                    })

            return future_value_debts, future_value_stocks, comparisons


        def create_visualizations(data: Dict) -> Tuple[str, str]:
            """
            Generate visualizations for future values of debts and stocks.

            Args:
                data (Dict): A dictionary containing future values of debts and stocks.

            Returns:
                Tuple[str, str]: A tuple containing two JSON strings for the visualizations of debt savings and stock returns.
            """
            future_values_debts = data['future_values']['future_value_debts']
            future_values_stocks = data['future_values']['future_value_stocks']
            fig_debts = go.Figure()
            fig_debts.add_trace(go.Bar(
                x=list(future_values_debts.keys()),
                y=list(future_values_debts.values()),
                text=[round(val, 2) for val in future_values_debts.values()],
                textposition='auto',
                name='Debts',
                marker_color='indianred'
            ))
            fig_debts.update_layout(
                title='Future Value of Debts',
                xaxis_title='Debts',
                yaxis_title='Future Value ($)',
                barmode='group'
            )

            fig_stocks = go.Figure()
            fig_stocks.add_trace(go.Bar(
                x=list(future_values_stocks.keys()),
                y=list(future_values_stocks.values()),
                text=[round(val, 2) for val in future_values_stocks.values()],
                textposition='auto',
                name='Stocks',
                marker_color='lightseagreen'
            ))
            fig_stocks.update_layout(
                title='Future Value of Stocks',
                xaxis_title='Stocks',
                yaxis_title='Future Value ($)',
                barmode='group'
            )
            debts_image_bytes = fig_debts.to_image(format="png")
            stocks_image_bytes = fig_stocks.to_image(format="png")
            debts_image_url = upload_image_to_host(debts_image_bytes)
            stocks_image_url = upload_image_to_host(stocks_image_bytes)
            return debts_image_url, stocks_image_url 
            
        @tool
        def credit_impact_multiplier(debt_type_list, credit_score_impact_list, legal_ramifications_list, base_interest_rate_list):
            """
            Function to adjust the interest rate of a list of debts based on their types, impacts on credit score,
            and legal ramifications of missing payments.

            Parameters:
            - debt_type (list): List of debt types (e.g., ['Credit Card', 'Tax Loan', 'Consumer Debt']).
            - credit_score_impact_list (list): List of credit score impacts (e.g., ['high', 'medium', 'low']).
            - legal_ramifications_list (list): List of legal ramifications (e.g., ['severe', 'moderate', 'minor']).
            - base_interest_rate_list (list): List of base annual interest rates on the debts.

            Returns:
            - list: List of dictionaries with adjusted interest rates for each debt.
            """
            debt_weights = {
                'Credit Card': {'credit_score_weight': 0.7, 'legal_weight': 0.6},
                'Tax Loan': {'credit_score_weight': 0.6, 'legal_weight': 0.7},
                'Consumer Debt': {'credit_score_weight': 0.5, 'legal_weight': 0.5},
                'Student Debt': {'credit_score_weight': 0.4, 'legal_weight': 0.4},
                'Government Debt': {'credit_score_weight': 0.3, 'legal_weight': 0.8}
            }
            impact_values = {
                'high': 0.6,
                'medium': 0.45,
                'low': 0.2
            }
            legal_values = {
                'severe': 0.6,
                'moderate': 0.3,
                'minor': 0.1
            }
            results = []
            for debt_type, credit_score_impact, legal_ramifications, base_interest_rate in zip(debt_type_list, credit_score_impact_list, legal_ramifications_list, base_interest_rate_list):
                weights = debt_weights.get(debt_type, {'credit_score_weight': 0.5, 'legal_weight': 0.5})
                credit_score_value = impact_values.get(credit_score_impact.lower(), 0.0)
                legal_value = legal_values.get(legal_ramifications.lower(), 0.0)
                multiplier = (credit_score_value * weights['credit_score_weight']) + (legal_value * weights['legal_weight'])
                if base_interest_rate == 0:
                    adjusted_interest_rate = base_interest_rate * (1 + multiplier)
                else:
                    adjusted_interest_rate = base_interest_rate * (1 + multiplier / 2)
                results.append({"debt_type": debt_type, "adjusted_interest_rate": adjusted_interest_rate})
            return results            
            
        def debt_investment_optimizer(debts, available_funds, multiplier_scores, stock_data, risk_free_rate_annual=0.02):
            """
            Optimizes the allocation of available funds between debt repayment and stock investment across multiple debts.

            Args:
                debts (list of dict): Debts with 'debt_name', 'debt_amount', 'interest_rate', and 'min_payment'.
                available_funds (float): Total funds available.
                multiplier_scores (dict): Impact multipliers for each debt.
                stock_data (list of dict): A list of dictionaries representing various stock options available for investment.

            Returns:
                dict: Optimal allocation with 'debt_payments', 'investment_amount', 'total_wealth' and 'comparisons'.
            """ 
            optimal_payments = {
                'debt_payments': [],
                'investment_amount': 0,
                'total_wealth': 0,
                'stock_investments': [],
                'comparisons': [],
                'future_values': {
                    'future_value_debts': {},
                    'future_value_stocks': {}
                }
            }
            remaining_funds = available_funds
            highest_min_payment = max([debt['min_payment'] for debt in debts])
            monthly_payment = highest_min_payment * 1.25

            future_value_debts, future_value_stocks = opportunity_cost_comparison(debts, stock_data, risk_free_rate_annual)
                
            optimal_payments['future_values']['future_value_debts'] = future_value_debts
            optimal_payments['future_values']['future_value_stocks'] = future_value_stocks

            for debt in debts:
                debt_name = debt['debt_name']
                interest_rate = debt['interest_rate']
                multiplier_score = multiplier_scores.get(debt_name, 0.1)
                if interest_rate == 0:
                    interest_savings_percent = multiplier_score
                else:
                    interest_savings_percent = interest_rate + multiplier_score
                for stock in stock_data[:5]:
                    stock_name = stock['symbol']
                    stock_return_percent = float(stock['average_return']) * 100  
                    comparison = {
                        'debt_name': debt_name,
                        'debt_interest_rate': interest_rate,
                        'stock_name': stock_name,
                        'stock_return_percent': stock_return_percent,
                        'interest_savings_percent': interest_savings_percent
                    }
                    optimal_payments['comparisons'].append(comparison)
            optimal_payments['comparisons'].sort(
                key=lambda x: (float(x['stock_return_percent']) - float(x['interest_savings_percent'])),
                reverse=True
            )
            for comparison in optimal_payments['comparisons']:
                if remaining_funds <= 0:
                    break
                debt_name = comparison['debt_name']
                stock_name = comparison['stock_name']
                interest_savings_percent = comparison['interest_savings_percent']
                stock_return_percent = comparison['stock_return_percent']
                if stock_return_percent > interest_savings_percent:
                    allocation_amount = remaining_funds * (stock_return_percent / (stock_return_percent + interest_savings_percent))
                    optimal_payments['stock_investments'].append({'stock_name': stock_name, 'investment_amount': allocation_amount})
                    remaining_funds -= allocation_amount
                else:
                    allocation_amount = remaining_funds * (interest_savings_percent / (stock_return_percent + interest_savings_percent))
                    optimal_payments['debt_payments'].append({'debt_name': debt_name, 'payment_amount': allocation_amount})
                    remaining_funds -= allocation_amount

            return optimal_payments
        @tool 
        def structure_stock_data(raw_stock_data):
            """
            Structure raw stock data for further analysis.

            Args:
                raw_stock_data (list): List of raw stock data dictionaries.

            Returns:
                list: List of structured stock data dictionaries.
            """
            top_investments = raw_stock_data['top_investments']
            structured_stock_data = []

            for stock in top_investments:
                # Extracting monthly returns as a list of floats
                monthly_returns = [float(return_data["monthly_return"]) for return_data in stock["returns"]]

                # Structuring the data
                structured_stock = {
                    "symbol": stock["symbol"],
                    "average_return": float(stock["average_return"]),
                    "current_price": float(stock["current_price"]),
                    "monthly_return": monthly_returns
                }

                structured_stock_data.append(structured_stock)

            return structured_stock_data

        @tool
        async def initial_payment_allocator(debts, available_funds, multiplier_scores, stock_data):
            """
            Allocates minimum payments first, then uses the remaining funds to optimize debt repayment and stock investments.

            Args:
                debts (list of dict): Debts with 'debt_name', 'debt_amount', 'interest_rate', and 'min_payment'.
                available_funds (float): Total amount of money currently available for allocation towards debts and investments.
                multiplier_scores (dict): Dictionary mapping each debt_name to a multiplier that adjusts the interest rate based on external factors (e.g., credit score impact, legal ramifications).
                stock_data (list of dict): A list of dictionaries representing various stock options available for investment.

            Returns:
                 dict: Optimal allocation with 'debt_payments', 'stock_investments' and 'comparisons'.
            """
            debt_payments = {debt['debt_name']: float(debt.get('min_payment', 0)) for debt in debts}
            remaining_funds = float(available_funds) - sum(debt_payments.values())

            formatted_stock_data = []
            for stock in stock_data:
                formatted_stock = {
                    'symbol': stock['symbol'],
                    'average_return': float(stock['average_return']),
                    'current_price': float(stock['current_price']),
                    'monthly_return': [float(monthly_return) for monthly_return in stock['monthly_return']]
                }
                formatted_stock_data.append(formatted_stock)

            optimal_payments = await debt_investment_optimizer(debts, remaining_funds, multiplier_scores, formatted_stock_data)
            comparisons = optimal_payments['comparisons']

            for payment in optimal_payments['debt_payments']:
                debt_name = payment['debt_name']
                payment_amount = payment['payment_amount']
                if debt_name in debt_payments:
                    debt_payments[debt_name] += payment_amount
                else:
                    debt_payments[debt_name] = payment_amount
            stock_investments = {investment['stock_name']: investment['investment_amount'] for investment in optimal_payments['stock_investments']}

            comparisons = json.loads(json.dumps(comparisons))
            debt_chart, stock_chart = create_visualizations(comparisons)
            return {
                'debt_payments': [{'debt_name': name, 'payment_amount': amount} for name, amount in debt_payments.items()],
                'stock_investments': [{'stock_name': name, 'investment_amount': amount} for name, amount in stock_investments.items()],
                'comparisons': comparisons,
                'visualizations': {'debt_chart': debt_chart, 'stock_chart': stock_chart}
            }

        fetch_debt_str = json.dumps(fetch_debt)
        input_str = f"I have a balance of $2,000 and need an optimal payment strategy. Analyze my debts: {fetch_debt_str} and investments. Consider the top-performing stocks from Health Care and Information Technology sectors, and compare their investment returns with debt repayment benefits over a period of time."
    
        input_data = {
            "input": input_str,
            "sectors_of_interest": "Health Care, Information Technology, Financials, Energy",
            "agent_scratchpad": ""
            }
        try:
            openAI = await open_key.plaintext()
            logger.info(f"OpenAI key retrieved: {openAI}")

            llm = ChatOpenAI(model="gpt-3.5-turbo-16k", openai_api_key=openAI, temperature=0)
            logger.info("LLM initialized")
        except Exception as e:
            logger.error(f"Error retrieving OpenAI key: {e}")
            raise e
        try:
            simple_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", """You are an autonmous financial planner. Your goal is to analyze the given stock and debt data to suggest an optimal payment plan for debt repayment and stock purchases for a single month assuming a balance of $2000.

                Instructions:                
                1. **Classify Information**:
                - Based on the information available, classify the impact on the credit score as 'high', 'medium', or 'low' and the legal ramifications as 'severe', 'moderate', or 'minor'.
                - These classifications should be derived from the textual information obtained from the search results.

                2. **Adjust Interest Rates**:
                - Use the `credit_impact_multiplier` tool to adjust the interest rate for each debt based on the classifications and the base interest rate.
                - The adjusted interest rate will reflect the combined impact on the credit score and legal ramifications.
                - Return JSON in a single line without whitespaces.

                3. **Retrieve Stock Data**:
                - Use the `fetch_stocks` tool to get performance data for top stocks in Health Care, Information Technology, Financials, and Energy sectors.
                - Ensure the stock data uses the key `symbol` for stock identifiers. Record the average returns and current prices of these top stocks.
                - Label the fetched data as `stock_data`.
                - Return JSON in a single line without whitespaces.
                         
                4. **Structure Stock Data**:
                - Use the `structure_stock_data` tool to ensure the `stock_data` is correctly formatted with `monthly_return` as a list.
                - Pass the `stock_data` to the `structure_stock_data` tool.
                - Return JSON in a single line without whitespaces.

                5. **Allocate Initial Payments**:
                - Use the `initial_payment_allocator` tool to allocate the minimum payments first for all debts.
                - For debts without specified minimum payments, set the payment between 5% and 15% of the debt amount, based on expected stock returns.
                - Pass the following data to the `initial_payment_allocator` tool:
                    - `debts`: List of dictionaries with 'debt_name', 'debt_amount', 'interest_rate', and 'min_payment'.
                    - `available_funds`: The total amount of money available for allocation.
                    - `multiplier_scores`: Dictionary mapping each debt_name to a multiplier that adjusts the interest rate based on credit score impact and legal ramifications.
                    - `stock_data`: List of dictionaries representing various stock options available for investment, each containing 'symbol', 'average_return', 'current_price', and 'monthly_return'.
                - Within the `initial_payment_allocator` tool, the `debt_investment_optimizer` function is used to compare the returns from stock investments with the savings from additional debt repayments and dynamically allocate remaining funds to either high-interest debts or high-return investments based on this comparison. This ensures the total stays within the budget.
                **- Ensure that `comparisons` is a list of dictionaries where each dictionary has the following keys: `debt_name`, `debt_interest_rate`, `stock_name`, `stock_return_percent`, `future_value_debt`, `future_value_stock`, and `interest_savings_percent`.**
                - Return JSON in a single line without whitespaces.
                         
                6. **Make Recommendations**:
                - Based on the results from the `initial_payment_allocator` and `debt_investment_optimizer` tools, determine the specific amounts to allocate towards each debt and stock investment. List two top-performing stocks to invest in and specify the exact amount to invest in each, including the average return, current price, and projected value of the investment over 2 years.
                - Prioritize investing in high performing stocks and increase debt payments where interest rates are higher than return rates from stocks.
                - Provide a detailed analysis with specific amounts to allocate towards each debt and how much to invest in stocks. List 1-3 top-performing stocks to invest in and specify the exact amount to invest in each.
                - Ensure the minimum required payments for each debt are met before allocating any additional funds. Allocate at least 10% of the available funds towards stock investments.
                - Indicate if partial payments are made for lower-priority debts due to budget limitations.
                - Return JSON in a single line without whitespaces.

                         
                7. **Structure the Output**:
                - Format the final decision in a JSON structure with the following fields:
                - `debts`: List of dictionaries with each debt's `debt_name`, `amount_to_be_paid`, and `current_total`.
                - `investments`: List of dictionaries with each investment's `symbol`, `average_return`, `current_price`, and `projected_value_in_2_years`.
                - `visualizations`: Include the URLs of the uploaded visualization images in the response.
                - `rationale`: Brief explanation of the quantifiable reasons for the payment plan, not exceeding three lines.               
                - Exclude any extraneous information and keep the response focused on actionable items.
                - Return JSON in a single line without whitespaces.
                """),
                        MessagesPlaceholder("chat_history", optional=True),
                        ("human", "{input}"),
                        MessagesPlaceholder("agent_scratchpad")
                    ]
                )
            toolkit = [fetch_stocks, credit_impact_multiplier, initial_payment_allocator, structure_stock_data] 
            agent = create_openai_tools_agent(llm, toolkit, simple_prompt_template)
            agent_executor = AgentExecutor(
                                            agent=agent, 
                                            tools=toolkit, 
                                            verbose=True,
                                            handle_parsing_errors=True,
                                            max_iterations=100,
                                            max_execution_time=180,
                                            return_intermediate_steps=True,
                            )

            logger.info("AgentExecutor initialized")
        except Exception as e:
            logger.error(f"Error initializing AgentExecutor: {e}")
            raise e
        
        try:
            set_debug(True)
            result = await agent_executor.ainvoke(input_data)
            logger.info(f"Final decision: {result}")
            return {"decision": result, "agent_scratchpad": input_data["agent_scratchpad"]}
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise e
        