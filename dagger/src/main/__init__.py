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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@object_type
class DebtRepayment:
    @function
    async def fetch_data(self, apiKey: Secret, sheet: Secret, sheet_two: Secret, open_key: Secret, fred_str: Secret, google_key: Secret, search_engine_id: Secret, name: str) -> str:
        """Returns a container that echoes whatever string argument is provided"""
        REDIS_HOST = 'guided-sawfish-54202.upstash.io'
        REDIS_PORT = 6379
        REDIS_USERNAME = 'default'
        REDIS_PASSWORD = 'AdO6AAIncDE4ZjM1ZTQ1ZTg5NjQ0YzhiYjRjM2Y2ZWNhNzA3ZDdhYXAxNTQyMDI'
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            username=REDIS_USERNAME,
            password=REDIS_PASSWORD,
            ssl=True  
        )

        fetch_balance = await dag.fetch_spreadsheet_data().fetch_data(apiKey, sheet, 'Edited')
        logger.info(f"balance_resp: {fetch_balance}")
        fetch_balance = self.convert_amounts(fetch_balance)
        fetch_debt = await dag.fetch_spreadsheet_data().fetch_data(apiKey, sheet_two, name)
        fetch_debt = self.restructure_data(fetch_debt)
        decision = await self.run_agent(open_key, google_key, search_engine_id, fred_str, fetch_debt, redis_client)        
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
        """Process balance data to extract specific columns and return as JSON string."""
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
        """Remove dollar sign and commas from amount and convert to float."""
        clean_amount = amount.replace('$', '').replace(',', '')
        return float(clean_amount)
    
    def convert_amounts(self, data: str) -> str:
        """Convert amounts in the fetched balance data to floats and return as JSON string."""
        data_json = json.loads(data)
        if not data_json:
            return json.dumps([])
        for row in data_json:
            if 'Amount' in row:
                row['Amount'] = self.clean_amount(row['Amount'])
        return json.dumps(data_json)
    
    async def run_agent(self, open_key, google_key, search_engine_id, fred_str, fetch_debt, redis_client) -> str:

            def upload_image_to_host(image_bytes):
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
                """Fetches stock data for given sectors."""
                stocks_data = await dag.get_stocks().stocks(sectors_of_interest)
                return json.dumps(stocks_data)
            
            async def calculate_time_value(items: List[Dict[str, str]], period: int) -> str:
                """Calculate the future value of an amount of money over a period with a given rate."""
                results = []
                periods = list(range(1, period + 1))
                for item in items:
                    cleaned_amount = re.sub(r'[$]', '', item['amount'])
                    future_value = await dag.calculate_time_value().calculate(period, cleaned_amount, item['rate'], fred_str)
                    results.append({
                        'type': item['type'],
                        'name': item['name'],
                        'future_value': future_value,
                        'periods': periods
                    })
                return json.dumps(results)
            
            def create_visualizations(comparisons: List[Dict]) -> Tuple[str, str]:
                """
                Generate visualizations for comparisons.

                Args:
                    comparisons (List[Dict]): A list of dictionaries where each dictionary contains:
                        - 'debt_name': Name of the debt.
                        - 'interest_savings_percent': Percentage of interest savings.
                        - 'stock_name': Name of the stock.
                        - 'stock_return_percent': Percentage of stock return.
                        - 'future_value_debt': Future value of the debt at different time periods.
                        - 'future_value_stock': Future value of the stock at different time periods.

                Returns:
                    Tuple[str, str]: A tuple containing two JSON strings for the visualizations of debt savings and stock returns.

                The AI should pass the comparison data to this tool. The tool will generate visualizations 
                that show how much $100 would result in terms of interest savings for each debt and what 
                the same amount would result in with different stock investments. The visualizations will 
                be returned as JSON strings.
                """
                fig = go.Figure()

                for comparison in comparisons:
                    debt_name = comparison['debt_name']
                    stock_name = comparison['stock_name']

                    future_value_debt = comparison['future_value_debt']
                    future_value_stock = comparison['future_value_stock']
                    periods = comparison['periods']

                    debt_values = [entry for entry in future_value_debt]  
                    stock_values = [entry for entry in future_value_stock]  

                    fig.add_trace(go.Scatter(x=periods, y=debt_values, mode='lines+markers', name=f'Debt: {debt_name}'))
                    fig.add_trace(go.Scatter(x=periods, y=stock_values, mode='lines+markers', name=f'Stock: {stock_name}'))

                fig.update_layout(
                    title='Future Value Comparison of Debts and Stocks',
                    xaxis_title='Time Periods',
                    yaxis_title='Future Value ($)',
                    legend_title='Investments'
                )

                debt_image_bytes = fig.to_image(format="png")

                debt_image_url = upload_image_to_host(debt_image_bytes)

                return debt_image_url, debt_image_url  

            
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
            
            async def debt_investment_optimizer(debts, available_funds, multiplier_scores, stock_data):
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
                    'comparisons': []
                }
                remaining_funds = available_funds
                increment=100
                time_period=5
                items_to_calculate = []

                for debt in debts:
                    debt_name = debt['debt_name']
                    interest_rate = debt['interest_rate']
                    items_to_calculate.append({
                        'type': 'debt',
                        'name': debt_name,
                        'amount': str(increment),
                        'rate': str(interest_rate)
                    })
                
                for stock in stock_data[:5]:
                    stock_name = stock['symbol']
                    stock_return = stock['average_return']
                    items_to_calculate.append({
                        'type': 'stock',
                        'name': stock_name,
                        'amount': str(increment),
                        'rate': str(stock_return)
                    })
                
                future_values_json = await calculate_time_value(items_to_calculate, time_period)
                future_values = json.loads(future_values_json)

                future_value_debts = {fv['name']: {'values': fv['future_value'], 'periods': fv['periods']} for fv in future_values if fv['type'] == 'debt'}
                future_value_stocks = {fv['name']: {'values': fv['future_value'], 'periods': fv['periods']} for fv in future_values if fv['type'] == 'stock'}

                for debt in debts:
                    debt_name = debt['debt_name']
                    interest_rate = debt['interest_rate']
                    multiplier_score = multiplier_scores.get(debt_name, 0.1)
                    if interest_rate == 0:
                        interest_savings_percent = multiplier_score
                    else:
                        interest_savings_percent = interest_rate + multiplier_score
                    
                    future_value_debt = future_value_debts[debt_name]['values']
                    periods = future_value_debts[debt_name]['periods']

                    for stock in stock_data[:5]:
                        stock_name = stock['symbol']
                        stock_return_percent = stock['average_return']
                        future_value_stock = future_value_stocks[stock_name]['values']

                        comparison = {
                            'debt_name': debt_name,
                            'debt_interest_rate': interest_rate,
                            'stock_name': stock_name,
                            'stock_return_percent': stock_return_percent,
                            'future_value_debt': json.dumps(future_value_debt),
                            'future_value_stock': json.dumps(future_value_stock),
                            'interest_savings_percent': interest_savings_percent,
                            'periods': periods
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
                        for payment in optimal_payments['debt_payments']:
                            if payment['debt_name'] == debt_name:
                                payment['payment_amount'] += allocation_amount
                                remaining_funds -= allocation_amount

                return optimal_payments

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

                optimal_payments = await debt_investment_optimizer(debts, remaining_funds, multiplier_scores, stock_data)
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

                4. **Allocate Initial Payments**:
                - Use the `initial_payment_allocator` tool to allocate the minimum payments first for all debts.
                - For debts without specified minimum payments, set the payment between 5% and 15% of the debt amount, based on expected stock returns.
                - Pass the following data to the `initial_payment_allocator` tool:
                    - `debts`: List of dictionaries with 'debt_name', 'debt_amount', 'interest_rate', and 'min_payment'.
                    - `available_funds`: The total amount of money available for allocation.
                    - `multiplier_scores`: Dictionary mapping each debt_name to a multiplier that adjusts the interest rate based on credit score impact and legal ramifications.
                    - `stock_data`: List of dictionaries representing various stock options available for investment, each containing 'name' and 'average_return'.
                - Within the `initial_payment_allocator` tool, the `debt_investment_optimizer` function is used to compare the returns from stock investments with the savings from additional debt repayments and dynamically allocate remaining funds to either high-interest debts or high-return investments based on this comparison. This ensures the total stays within the budget.
                **- Ensure that `comparisons` is a list of dictionaries where each dictionary has the following keys: `debt_name`, `debt_interest_rate`, `stock_name`, `stock_return_percent`, `future_value_debt`, `future_value_stock`, and `interest_savings_percent`.**
                - Return JSON in a single line without whitespaces.
                         
                5. **Make Recommendations**:
                - Based on the results from the `initial_payment_allocator` and `debt_investment_optimizer` tools, determine the specific amounts to allocate towards each debt and stock investment. List two top-performing stocks to invest in and specify the exact amount to invest in each, including the average return, current price, and projected value of the investment over 2 years.
                - Prioritize investing in high performing stocks and increase debt payments where interest rates are higher than return rates from stocks.
                - Provide a detailed analysis with specific amounts to allocate towards each debt and how much to invest in stocks. List 1-3 top-performing stocks to invest in and specify the exact amount to invest in each.
                - Ensure the minimum required payments for each debt are met before allocating any additional funds. Allocate at least 10% of the available funds towards stock investments.
                - Indicate if partial payments are made for lower-priority debts due to budget limitations.
                - Return JSON in a single line without whitespaces.

                         
                6. **Structure the Output**:
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
                toolkit = [fetch_stocks, credit_impact_multiplier, initial_payment_allocator] 
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
        