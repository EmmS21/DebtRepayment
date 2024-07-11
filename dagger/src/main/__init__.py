from dagger import dag, function, object_type, Secret
import logging
import json
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re
import redis
import requests
import plotly.graph_objects as go
from typing import Dict, Tuple
import base64
from datetime import datetime
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@object_type
class DebtRepayment:
    @function
    async def fetch_data(self, apiKey: Secret, sheet: Secret, sheet_two: Secret, open_key: Secret, send_grid: Secret, name: str, send_to: str, email: str, imgrKey: Secret) -> dict[str, str]:
        """
        Fetch data from two Google Sheets and run an agent to analyze the debts and investments.
        
        Args:
            apiKey (Secret): API key for accessing the Google Sheets API.
            sheet (Secret): Secret for the first Google Sheet.
            sheet_two (Secret): Secret for the second Google Sheet.
            open_key (Secret): API key for OpenAI.
            name (str): Name of the sheet to fetch data from.
            send_to (str): Email to send output to
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

        def find_json(s):
            bracket_count = 0
            start = None
            for i, char in enumerate(s):
                if char == '{':
                    if bracket_count == 0:
                        start = i
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0 and start is not None:
                        return s[start:i+1]
            return None

        json_str = find_json(output)
        clean_data = None
        if json_str:
            try:
                json_str = re.sub(r"(?<!\\)'([^']+)'(?=\s*:)", r'"\1"', json_str)
                json_str = re.sub(r':\s*\'([^\']+)\'', r': "\1"', json_str)
                
                clean_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON object: {e}")  

        if clean_data is None:
            print("Error: No valid JSON objects found")
            return None        
        return await self.send_email(send_grid, 'Emmanuel from GetStocked', email, send_to, json.dumps(clean_data))
        
    
    async def send_email(self, send_grid: Secret, sender_name: str, sender_email: str, recipient_email: str, clean_decision: dict) -> None:
        """
        Send an email campaign using Brevo's API.

        Args:
            sender_name (str): Name of the email sender.
            sender_email (str): Email address of the sender.
            recipient_email (str): Email address of the recipient.
            clean_decision (dict): The decision data to include in the email.

        Returns:
            None
        """
        if not isinstance(clean_decision, str):
            clean_decision = json.dumps(clean_decision)

        cleaned_data = None
        for line in clean_decision.splitlines():
            cleaned_data = json.loads(line)
            break  

        if not cleaned_data:
            print("Error: No valid JSON object found")
            return None
            
        api_key = await send_grid.plaintext()
        now = datetime.now()
        current_month = now.strftime("%B")
        current_year = now.year
        email_subject = f"Your Debt Repayment Plan for {current_month} {current_year}"
        email_body = f"<h2>Here is your repayment plan</h2>"

        for debt in cleaned_data["debts"]:
            email_body += f"<p>{debt['debt_name']}: ${debt['amount_to_be_paid']}</p>"
        email_body += f"<p><strong>Rationale:</strong> {cleaned_data['rationale']}</p>"
        email_body += f"<img src='{cleaned_data['visualizations']['debt_chart']}' alt='Debt Chart'>"
        email_body += f"<img src='{cleaned_data['visualizations']['stock_chart']}' alt='Stock Chart'>"

        message = Mail(
            from_email=Email(sender_email, sender_name),
            to_emails=To(recipient_email),
            subject=email_subject,
            html_content=Content("text/html", email_body)
        )

        try:
            sg = sendgrid.SendGridAPIClient(api_key)
            response = sg.send(message)
            print(response.status_code)
            print(response.body)
            print(response.headers)
        except Exception as e:
            print(e)

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
    
    async def run_agent(self, open_key, fetch_debt, imgrKey) -> str:
        """
        Run an agent to analyze debts and investments and return a decision.

        Args:
            open_key (Secret): API key for OpenAI.
            fetch_debt (str): JSON string of fetched debt data.

        Returns:
            str: JSON string containing the decision from the agent.
        """
        imgrAPIKey = await imgrKey.plaintext()
        def upload_image_to_host(image_bytes):
            """
            Upload image to a hosting service and return the URL.

            Args:
                image_bytes (bytes): Image bytes to be uploaded.

            Returns:
                str: URL of the uploaded image.
            """
            api_endpoint = "https://api.imgbb.com/1/upload"
            api_key = imgrAPIKey
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
            stocks_data = await dag.get_stocks().stocks(sectors_of_interest)
            return json.dumps(stocks_data)
                        
        def opportunity_cost_comparison(debts, stock_data, multiplier_scores):
                """
                Compare opportunity costs between debts and stock investments.

                Args:
                    debts (list): List of debt dictionaries.
                    stock_data (list): List of stock data dictionaries.
                    multiplier_scores (dict): Multiplier scores for each debt.

                Returns:
                    tuple: Future value of debts, future value of stocks, and comparison results.
                """
                future_value_debts = {}
                future_value_stocks = {}
                comparisons = []

                for debt in debts:
                    debt_name = debt['debt_name']
                        
                    multiplier_score = multiplier_scores.get(debt_name, 0.1)
                    interest_rate = debt.get('interest_rate', 0)  
                    if interest_rate == 0:
                        interest_rate = (5 + (5 + multiplier_score / 100)) / 100 / 12 
                    else:
                        interest_rate = debt['interest_rate'] / 100 / 12
                    future_value_debt = 0
                    normalized_balance = 1000
                    future_value_debt = normalized_balance * interest_rate 
                    future_value_debts[debt_name] = future_value_debt

                for stock in stock_data:
                    stock_name = stock['symbol']
                    annual_return = float(stock['average_return'])
                    
                    future_value_stock = 0
                    normalized_investment = 1000
                    future_value_stock = normalized_investment * (annual_return/12)
                    future_value_stocks[stock_name] = future_value_stock

                    for debt_name, debt_value in future_value_debts.items():
                        debt_interest_saving = debt_value
                        stock_return = future_value_stock
                        comparisons.append({
                            'debt_name': debt_name,
                            'stock_name': stock_name,
                            'debt_interest_saving': debt_interest_saving,
                            'stock_return': stock_return,
                            'monthly_return': annual_return/12
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
            if isinstance(data, list):
                data = data[0]
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
            
        def debt_investment_optimizer(debts, available_funds, multiplier_scores, stock_data):
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

            for debt in debts:
                debt_name = debt['debt_name']
                min_payment = debt['min_payment']
                if min_payment > 0:
                    optimal_payments['debt_payments'].append({'debt_name': debt_name, 'payment_amount': min_payment})
                    remaining_funds -= min_payment

            future_value_debts, future_value_stocks, comparisons = opportunity_cost_comparison(debts, stock_data, multiplier_scores)
            optimal_payments['future_values']['future_value_debts'] = future_value_debts
            optimal_payments['future_values']['future_value_stocks'] = future_value_stocks
            optimal_payments['comparisons'] = comparisons
            top_stocks = sorted(future_value_stocks.items(), key=lambda x: x[1], reverse=True)[:2]
            total_top_stocks_value = sum([value for _, value in top_stocks])
            total_future_values = sum(future_value_debts.values()) + total_top_stocks_value
            
            for debt_name, future_value_debt in future_value_debts.items():
                proportion = future_value_debt / total_future_values
                allocation = remaining_funds * proportion
                optimal_payments['debt_payments'].append({'debt_name': debt_name, 'payment_amount': round(allocation, 2)})

            for stock_name, future_value_stock in top_stocks:
                proportion = future_value_stock / total_future_values
                allocation = remaining_funds * proportion
                optimal_payments['stock_investments'].append({'stock_name': stock_name, 'investment_amount': round(allocation, 2)})

            return optimal_payments

        @tool
        def initial_payment_allocator(debts, available_funds, multiplier_scores, stock_data):
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
            formatted_stock_data = []
            for stock in stock_data:
                formatted_stock = {
                    'symbol': stock['symbol'],
                    'average_return': float(stock['average_return']),
                    'current_price': float(stock['current_price'])
                }
                formatted_stock_data.append(formatted_stock)

            optimal_payments = debt_investment_optimizer(debts, available_funds, multiplier_scores, formatted_stock_data)
            comparisons = optimal_payments['comparisons']

            debt_payments = {}
            for payment in optimal_payments['debt_payments']:
                debt_name = payment['debt_name']
                payment_amount = payment['payment_amount']
                if debt_name in debt_payments:
                    debt_payments[debt_name] += payment_amount
                else:
                    debt_payments[debt_name] = payment_amount

            stock_investments = {investment['stock_name']: investment['investment_amount'] for investment in optimal_payments['stock_investments']}

            comparisons = json.loads(json.dumps(comparisons))
            future_value_debts = optimal_payments['future_values']['future_value_debts']
            future_value_stocks = optimal_payments['future_values']['future_value_stocks']
            data = {
                'future_values': {
                    'future_value_debts': future_value_debts,
                    'future_value_stocks': future_value_stocks
                }
            }
            debt_chart, stock_chart = create_visualizations(data)

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
                - Use the `initial_payment_allocator` tool to allocate  money towards debt payment and stock investment.
                - For debts without specified minimum payments, set the payment between 5% and 15% of the debt amount, based on expected stock returns.
                - Pass the following data to the `initial_payment_allocator` tool:
                    - `debts`: List of dictionaries with 'debt_name', 'debt_amount', 'interest_rate', and 'min_payment'.
                    - `available_funds`: The total amount of money available for allocation.
                    - `multiplier_scores`: Dictionary mapping each debt_name to a multiplier that adjusts the interest rate based on credit score impact and legal ramifications.
                    - `stock_data`: List of dictionaries representing various stock options available for investment, each containing 'symbol', 'average_return', 'current_price'.
                - Within the `initial_payment_allocator` tool, the `debt_investment_optimizer` function is used to compare the returns from stock investments with the savings from additional debt repayments and dynamically allocate remaining funds to either high-interest debts or high-return investments based on this comparison. This ensures the total stays within the budget.
                **- Ensure that `comparisons` is a list of dictionaries where each dictionary has the following keys: `debt_name`, `debt_interest_rate`, `stock_name`, `stock_return_percent`, `future_value_debt`, `future_value_stock`, and `interest_savings_percent`.**
                - Return JSON in a single line without whitespaces.
                
                5. **Clear `top_investments`**:
                - After the returning the JSON from the `initial_payment_allocator` tool, clear the `stock_data` from the context, using the remaining data to explain 
                - This ensures the maximum context length is not exceeded.
                         
                6. **Make Recommendations**:
                - Use the results from the `initial_payment_allocator` and `debt_investment_optimizer` tools to return a result with debt repayment and stock investment allocations.
                - Return the amount to be paid for each debt based on the results from the `initial_payment_allocator` under the `debt_payments` key and the amount to be invested for each stock from the `stock_investments` key.
                - Provide a detailed analysis of the allocation using the allocation data received from the `initial_payment_allocator`, the graphs returned from this tool, and other contextual information you have access to.
                - Use the `comparisons` key for the rationale.
                - Explain things as simply as possible to help me clearly understand what the allocations mean and how they will help me.
                - Return JSON in a single line without whitespaces.

                         
                7. **Structure the Output**:
                - Format the final decision in a JSON structure with the following fields:
                - `debts`: List of dictionaries with each debt's `debt_name`, `amount_to_be_paid`, and `current_total`.
                - `investments`: List of dictionaries with each stock_investments's `stock name`, `investment_amount`, `current_price`, and `projected_value_in_2_years`.
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
        except Exception as e:
            raise e        
        try:
            result = await agent_executor.ainvoke(input_data)
            return {"decision": result, "agent_scratchpad": input_data["agent_scratchpad"]}
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise e
    
