from dagger import dag, function, object_type, Secret
import logging
import json
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SerpAPIWrapper
import re
import redis
import numpy as np
from langchain.globals import set_debug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@object_type
class DebtRepayment:
    @function
    async def fetch_data(self, apiKey: Secret, sheet: Secret, sheet_two: Secret, open_key: Secret, fred_str: Secret, serpapi_key: Secret, name: str) -> str:
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
        decision = await self.run_agent(open_key, serpapi_key, fred_str, fetch_debt, redis_client)        
        output = decision.get("decision", {}).get("output", "")
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
    
    async def calculate_time_value(self, period:int, amount: str, rate: str, fred_str: Secret):
        return dag.calculate_time_value().calculate(period, amount, rate, fred_str)
    
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
    
    async def run_agent(self, open_key, serpapi_key, fred_str, fetch_debt, redis_client) -> str:
            SERPAPI_API_KEY = await serpapi_key.plaintext()
            @tool
            def serp_search(query: str) -> str:
                """Searches for information using SerpAPI."""
                try:
                    cached_result = redis_client.get(query)
                    if cached_result:
                        return cached_result.decode('utf-8')
                    serp = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
                    response = serp.run(query)
                    if isinstance(response, (list, dict)):
                        response = json.dumps(response)
                    redis_client.set(query, response)
                    return response
                except Exception as e:
                    cached_result = redis_client.get(query)
                    if cached_result:
                        return cached_result.decode('utf-8')
                    else:
                        return "No cached result available"
            @tool  
            async def fetch_stocks(sectors_of_interest:str) -> str:
                """Fetches stock data for given sectors."""
                stocks_data = await dag.get_stocks().stocks(sectors_of_interest)
                return json.dumps(stocks_data)
            
            @tool
            async def calculate_time_value(period: int, amount: str, rate: str) -> str:
                """Calculate the future value of an amount of money over a period with a given rate."""
                future_value = await dag.calculate_time_value().calculate(period, amount, rate, fred_str)
                return json.dumps(future_value)

            @tool
            def credit_impact_multiplier(debt_type, credit_score_impact, legal_ramifications, base_interest_rate):
                """
                Function to adjust the interest rate of a debt based on its type, impact on credit score,
                and legal ramifications of missing payments.

                Parameters:
                - debt_type (str): The type of debt (e.g., 'Credit Card', 'Tax Loan', 'Consumer Debt', 'Student Debt', 'Government Debt').
                - credit_score_impact (str): The impact of this debt on the credit score (e.g., 'high', 'medium', 'low').
                - legal_ramifications (str): The legal consequences of missing payments for this debt (e.g., 'severe', 'moderate', 'minor').
                - base_interest_rate (float): The base annual interest rate on the debt.

                Returns:
                - adjusted_interest_rate (float): The adjusted interest rate based on the provided information.
                """
                debt_weights = {
                    'Credit Card': {'credit_score_weight': 0.7, 'legal_weight': 0.6},
                    'Tax Loan': {'credit_score_weight': 0.6, 'legal_weight': 0.7},
                    'Consumer Debt': {'credit_score_weight': 0.5, 'legal_weight': 0.5},
                    'Student Debt': {'credit_score_weight': 0.4, 'legal_weight': 0.4},
                    'Government Debt': {'credit_score_weight': 0.3, 'legal_weight': 0.8}
                }
                impact_values = {
                    'high': 0.8,
                    'medium': 0.5,
                    'low': 0.2
                }
                legal_values = {
                    'severe': 0.7,
                    'moderate': 0.4,
                    'minor': 0.1
                }
                weights = debt_weights.get(debt_type, {'credit_score_weight': 0.5, 'legal_weight': 0.5})
                credit_score_value = impact_values.get(credit_score_impact.lower(), 0.0)
                legal_value = legal_values.get(legal_ramifications.lower(), 0.0)
                multiplier = (credit_score_value * weights['credit_score_weight']) + (legal_value * weights['legal_weight'])
                if base_interest_rate == 0:
                    adjusted_interest_rate = base_interest_rate * (1 + multiplier)
                else:
                    adjusted_interest_rate = base_interest_rate * (1 + multiplier / 2)
                    
                return {"adjusted_interest_rate": adjusted_interest_rate}
            
            @tool
            async def validate_allocations(allocations: str, budget: float = 2000.0) -> str:
                """
                Validate that the total allocations do not exceed the budget.

                Args:
                    allocations (str): A JSON string representing a list of allocation dictionaries, each containing 'type', 'name', and 'amount'.
                    budget (float): The budget limit for allocations.

                Returns:
                    str: A JSON string containing the total allocated amount, whether it is within the budget, and the validated allocations.
                """
                try:
                    allocations_list = json.loads(allocations)
                    total_allocated = sum(item['amount'] for item in allocations_list)
                    is_within_budget = total_allocated <= budget

                    if not is_within_budget:
                        validated_allocations = []
                        total_allocated = 0.0

                        for allocation in allocations_list:
                            if total_allocated + allocation['amount'] <= budget:
                                validated_allocations.append(allocation)
                                total_allocated += allocation['amount']
                            else:
                                remaining_budget = budget - total_allocated
                                if remaining_budget > 0:
                                    allocation['amount'] = remaining_budget
                                    validated_allocations.append(allocation)
                                    total_allocated += remaining_budget
                                break

                        result = {
                            'total_allocated': total_allocated,
                            'is_within_budget': total_allocated <= budget,
                            'validated_allocations': validated_allocations
                        }
                    else:
                        result = {
                            'total_allocated': total_allocated,
                            'is_within_budget': is_within_budget,
                            'validated_allocations': allocations_list
                        }

                    return json.dumps(result)
                except json.JSONDecodeError as e:
                    return json.dumps({"error": "JSON decoding error", "details": str(e), "input": allocations})
                except Exception as e:
                    return json.dumps({"error": "An error occurred", "details": str(e), "input": allocations})

            def debt_investment_optimizer(debt_amount, interest_rate, expected_stock_return, multiplier_score, available_funds, min_payment):
                """
                Optimizes the allocation of available funds between debt repayment and stock investment.

                Args:
                    debt_amount (float): The outstanding debt amount.
                    interest_rate (float): The annual interest rate on the debt.
                    expected_stock_return (float): The expected annual rate of return on the stock investment.
                    multiplier_score (float): A value representing the impact of debt on factors like credit score.
                    available_funds (float): The total amount available for debt repayment and stock investment.
                    min_payment (float): The minimum required payment for the debt.

                Returns:
                    dict: A dictionary with the optimal allocation, including:
                        - 'debt_payment_amount': Amount allocated towards debt repayment.
                        - 'investment_amount': Amount allocated to stock investment.
                        - 'debt_repayment_period': Number of years to repay the debt.
                        - 'total_wealth': Total wealth after the debt repayment period, considering debt and investment returns.
                """
                min_payment = max(min_payment, 50)  
                optimal_payment = min_payment
                max_wealth = -float('inf')

                def total_wealth(payment_amount):
                    if payment_amount == 0:
                        return -float('inf')

                    debt_repayment_period = np.ceil(debt_amount / payment_amount)
                    investment_amount = available_funds - payment_amount
                    max_period = min(debt_repayment_period, 100)
                    stock_returns = investment_amount * (1 + expected_stock_return) ** max_period

                    if interest_rate == 0:
                        remaining_debt = debt_amount * (1 + multiplier_score) 
                    else:
                        remaining_debt = debt_amount * (1 + interest_rate) ** max_period * (1 + multiplier_score)
                    
                    return stock_returns - remaining_debt

                for payment_amount in np.arange(min_payment, available_funds + 1, 20): 
                    wealth = total_wealth(payment_amount)
                    if wealth > max_wealth:
                        max_wealth = wealth
                        optimal_payment = payment_amount

                debt_repayment_period = np.ceil(debt_amount / optimal_payment)
                investment_amount = available_funds - optimal_payment

                return json.dumps({
                    "debt_payment_amount": optimal_payment,
                    "investment_amount": investment_amount,
                    "debt_repayment_period": int(debt_repayment_period),
                    "total_wealth": max_wealth
                })

            @tool
            def initial_payment_allocator(debts, available_funds, expected_stock_return, multiplier_scores):
                """
                Allocates minimum payments first, then uses the remaining funds to optimize debt repayment and stock investments.

                Args:
                    debts (list of dict): Debts with 'debt_name', 'debt_amount', 'interest_rate', and 'min_payment'.
                    available_funds (float): Total funds available.
                    expected_stock_return (float): Expected annual return on stocks.
                    multiplier_scores (dict): Impact multipliers for each debt.

                Returns:
                    dict: Optimal allocation with 'debt_payments', 'investment_amount', and 'total_wealth'.
                """
                debt_payments = []
                total_min_payments = 0

                print(f"Debts: {debts}")
                print(f"Available Funds: {available_funds}")
                print(f"Expected Stock Return: {expected_stock_return}")
                print(f"Multiplier Scores: {multiplier_scores}")

                # Calculate total minimum payments
                for debt in debts:
                    min_payment = debt.get('min_payment', 50) 
                    if min_payment is None or min_payment == 0:
                        min_payment = 50
                    if isinstance(min_payment, (int, float)):
                        total_min_payments += min_payment
                    else:
                        raise ValueError(f"Invalid min_payment for debt {debt['debt_name']}: {min_payment}")

                # Ensure available funds cover the minimum payments
                if available_funds < total_min_payments:
                    raise ValueError("Available funds are less than the total minimum payments required")

                remaining_funds = available_funds - total_min_payments
                
                # Distribute minimum payments
                for debt in debts:
                    min_payment = debt.get('min_payment', 50)  
                    if min_payment is None or min_payment == 0:
                        min_payment = 50
                    debt_payments.append({'debt_name': debt['debt_name'], 'payment_amount': min_payment})
                
                print(f"Debt Payments: {debt_payments}")
                print(f"Remaining Funds: {remaining_funds}")

                optimal_payments = {
                    'debt_payments': debt_payments,
                    'investment_amount': 0,
                    'total_wealth': -float('inf')
                }
                max_wealth = -float('inf')
                
                for debt in debts:
                    debt_name = debt['debt_name']
                    debt_amount = debt['debt_amount']
                    interest_rate = debt['interest_rate']
                    min_payment = debt.get('min_payment', 50)  
                    if min_payment is None or min_payment == 0:
                        min_payment = 50
                    multiplier_score = multiplier_scores.get(debt_name, 0.1) 

                    if not isinstance(min_payment, (int, float)):
                        raise ValueError(f"Invalid min_payment for debt {debt_name}: {min_payment}")
                    
                    result_json = debt_investment_optimizer(debt_amount, interest_rate, expected_stock_return, multiplier_score, remaining_funds, min_payment)
                    result = json.loads(result_json)

                    print(f"Result for {debt_name}: {result}")
                      
                    if result['total_wealth'] > max_wealth:
                        max_wealth = result['total_wealth']
                        optimal_payments = {
                            'debt_payments': debt_payments + [{'debt_name': debt_name, 'payment_amount': result['debt_payment_amount']}],
                            'investment_amount': result['investment_amount'],
                            'total_wealth': result['total_wealth']
                        }
                        
                print(f"Optimal Payments: {optimal_payments}")
                return optimal_payments

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

                llm = ChatOpenAI(model="gpt-3.5-turbo-16k", openai_api_key=openAI)
                logger.info("LLM initialized")
            except Exception as e:
                logger.error(f"Error retrieving OpenAI key: {e}")
                raise e
            try:
                simple_prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", """You are an autonmous financial planner. Your goal is to analyze the given stock and debt data to suggest an optimal payment plan for debt repayment and stock purchases for a single month assuming a balance of $2000.

                Instructions:
                1. **Search Debt Information**:
                - Use the `serp_search` tool with SerpAPI to find detailed information about each debt type.
                - Gather contextual information that helps determine the impact of each debt on the credit score, legal ramifications of missing payments, and other relevant factors.

                2. **Classify Information**:
                - Based on the gathered information, classify the impact on the credit score as 'high', 'medium', or 'low' and the legal ramifications as 'severe', 'moderate', or 'minor'.
                - These classifications should be derived from the textual information obtained from the search results.

                3. **Adjust Interest Rates**:
                - Use the `credit_impact_multiplier` tool to adjust the interest rate for each debt based on the classifications and the base interest rate.
                - The adjusted interest rate will reflect the combined impact on the credit score and legal ramifications.

                4. **Retrieve Stock Data**:
                - Use the `fetch_stocks` tool to get performance data for top stocks in Health Care, Information Technology, Financials, and Energy sectors.
                - Record the average returns and current prices of these top stocks.

                5. **Calculate Future Values**:
                - Use the `calculate_time_value` tool to calculate the future values of potential stock investments over a sensible time period (e.g., maximum time to pay back debt).
                - Calculate the future value of each debt given the interest rates.

                6. **Allocate Initial Payments**:
                - Use the `initial_payment_allocator` tool to allocate the minimum payments first and use the remaining funds to maximize wealth.
                - Pass the list of debts, available budget ($2000), expected stock return, and multiplier scores as arguments to the tool.
                - The tool will return the optimal debt payment amounts, investment amount, and total wealth.

                7. **Make Recommendations**:
                - Based on the results from the `initial_payment_allocator` tool, determine the specific amounts to allocate towards each debt and stock investment.
                - Provide a detailed analysis with specific amounts to allocate towards each debt and how much to invest in stocks. List 1-3 top-performing stocks to invest in and specify the exact amount to invest in each.
                - Ensure the minimum required payments for each debt are met before allocating any additional funds.
                - Indicate if partial payments are made for lower-priority debts due to budget limitations.

                Output:
                - Return a concise JSON with the following:
                1. Each debt and the exact amount to be paid off this month.
                2. Stocks to invest in, the exact amount to invest, and the projected return over the chosen period.
                3. A brief rationale explaining the decisions and quantifiable benefits.
                Exclude any extraneous information and keep the response focused on actionable items.
                Only include necessary details in the response.
                """),
                        MessagesPlaceholder("chat_history", optional=True),
                        ("human", "{input}"),
                        MessagesPlaceholder("agent_scratchpad")
                    ]
                )
                toolkit = [fetch_stocks, calculate_time_value, serp_search, credit_impact_multiplier, initial_payment_allocator, validate_allocations] 
                agent = create_openai_tools_agent(llm, toolkit, simple_prompt_template)
                agent_executor = AgentExecutor(
                                               agent=agent, 
                                               tools=toolkit, 
                                               verbose=True,
                                               handle_parsing_errors=True,
                                               max_iterations=70,
                                               max_execution_time=180
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
        



# he calculated weighting of importance for each debt, comparing the time value of the debt compared with different stock investments and any other considerations relevant.
