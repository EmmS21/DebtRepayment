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
from itertools import product

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
                If the total exceeds the budget, distribute the remaining budget proportionally among the allocations.

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
                        remaining_budget = budget
                        validated_allocations = []
                        for allocation in allocations_list:
                            allocation_share = allocation['amount'] / total_allocated * remaining_budget
                            allocation['amount'] = allocation_share
                            validated_allocations.append(allocation)
                            remaining_budget -= allocation_share

                    result = {
                        'total_allocated': total_allocated,
                        'is_within_budget': is_within_budget,
                        'validated_allocations': validated_allocations
                    }

                    return json.dumps(result)
                except json.JSONDecodeError as e:
                    return json.dumps({"error": "JSON decoding error", "details": str(e), "input": allocations})
                except Exception as e:
                    return json.dumps({"error": "An error occurred", "details": str(e), "input": allocations})

            def debt_investment_optimizer(debts, available_funds, expected_stock_return, multiplier_scores, stock_data):
                """
                Optimizes the allocation of available funds between debt repayment and stock investment across multiple debts.

                Args:
                    debts (list of dict): Debts with 'debt_name', 'debt_amount', 'interest_rate', and 'min_payment'.
                    available_funds (float): Total funds available.
                    expected_stock_return (float): Expected annual rate of return on stocks.
                    multiplier_scores (dict): Impact multipliers for each debt.

                Returns:
                    dict: Optimal allocation with 'debt_payments', 'investment_amount', and 'total_wealth'.
                """
                max_wealth = -float('inf')
                optimal_payments = {
                    'debt_payments': [],
                    'investment_amount': 0,
                    'total_wealth': -float('inf'),
                }

                def total_wealth(payment_amounts):
                    remaining_funds = available_funds
                    investment_amount = max(0.1 * available_funds, available_funds - sum(payment_amounts))
                    total_remaining_debt = 0
                    for debt, payment_amount in zip(debts, payment_amounts):
                        debt_amount = debt['debt_amount']
                        interest_rate = debt['interest_rate']
                        multiplier_score = multiplier_scores.get(debt['debt_name'], 0.1)
                        min_payment = max(payment_amount, debt.get('min_payment', 0))
                        remaining_funds -= min_payment
                        if interest_rate == 0:
                            remaining_debt = debt_amount * (1 + multiplier_score)
                        else:
                            remaining_debt = debt_amount * (1 + interest_rate) ** 100 * (1 + multiplier_score)
                        total_remaining_debt += remaining_debt
                    stock_returns = investment_amount * (1 + expected_stock_return) ** 100
                    total_wealth = stock_returns - total_remaining_debt
                    return total_wealth, investment_amount

                max_payment = int(available_funds / len(debts))
                payment_ranges = [range(int(debt.get('min_payment', 0)), max_payment + 1, 20) for debt in debts]
                for payment_amounts in product(*payment_ranges):
                    wealth, investment_amount = total_wealth(payment_amounts)
                    if wealth > max_wealth:
                        max_wealth = wealth
                        optimal_payments = {
                            'debt_payments': [{'debt_name': debt['debt_name'], 'payment_amount': amount}
                                            for debt, amount in zip(debts, payment_amounts)],
                            'investment_amount': investment_amount,
                            'total_wealth': wealth,
                            'stock_investments': []
                        }

                if optimal_payments['investment_amount'] > 0:
                    total_return = sum(stock['average_return'] for stock in stock_data)
                    optimal_investments = [{'stock_name': stock['name'], 'investment_amount': (stock['average_return'] / total_return) * optimal_payments['investment_amount']} for stock in stock_data]
                    optimal_payments['stock_investments'] = optimal_investments

                return optimal_payments

            @tool
            def initial_payment_allocator(debts, available_funds, expected_stock_return, multiplier_scores, stock_data):
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
                for debt in debts:
                    min_payment = debt.get('min_payment', None)
                    debt_amount = debt['debt_amount']
                    if min_payment is None or min_payment == 0:
                        if debt_amount > 10000:
                            min_payment_percentage = 0.01 + (0.03 * (1 - expected_stock_return / 0.2))
                            min_payment_percentage = max(0.01, min(0.04, min_payment_percentage))
                        else:
                            min_payment_percentage = 0.02 + (0.08 * (1 - expected_stock_return / 0.2))
                            min_payment_percentage = max(0.02, min(0.1, min_payment_percentage))
                        min_payment = min_payment_percentage * debt_amount
                    debt_payments.append({'debt_name': debt['debt_name'], 'payment_amount': min_payment})

                remaining_funds = available_funds - sum(payment['payment_amount'] for payment in debt_payments)

                optimal_payments = debt_investment_optimizer(debts, remaining_funds, expected_stock_return, multiplier_scores, stock_data)
                optimal_payments['debt_payments'] += debt_payments

                total_allocated = sum(payment['payment_amount'] for payment in optimal_payments['debt_payments']) + optimal_payments['investment_amount']
                if total_allocated > available_funds:
                    remaining_budget = available_funds
                    adjusted_payments = []
                    sorted_debts = sorted(debts, key=lambda d: d['interest_rate'] * (1 + multiplier_scores.get(d['debt_name'], 0.1)), reverse=True)
                    for debt in sorted_debts:
                        debt_name = debt['debt_name']
                        payment_amount = next((p['payment_amount'] for p in optimal_payments['debt_payments'] if p['debt_name'] == debt_name), 0)
                        if remaining_budget >= payment_amount:
                            adjusted_payments.append({'debt_name': debt_name, 'payment_amount': payment_amount})
                            remaining_budget -= payment_amount
                        else:
                            adjusted_payments.append({'debt_name': debt_name, 'payment_amount': remaining_budget})
                            remaining_budget = 0
                            break
                    optimal_payments['debt_payments'] = adjusted_payments
                    optimal_payments['investment_amount'] = remaining_budget

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
                - The minimum payment for debts with no specified minimum will be set between 5% and 15% of the debt amount, based on the expected stock returns.
                - Pass the list of debts, available budget ($2000), expected stock return, and multiplier scores as arguments to the tool.
                - The tool will use the `debt_investment_optimizer` function to determine the optimal debt payment amounts and investment amount that maximize the total wealth.
                - At least 10% of the available funds should be allocated towards stock investments, unless the expected stock returns are significantly lower than the debt interest rates.
                - The allocations should not exceed the available budget of $2000.

                7. **Make Recommendations**:
                - Based on the results from the `initial_payment_allocator` tool, determine the specific amounts to allocate towards each debt and stock investment.
                - Prioritize growing net worth while maintaining a good credit score.
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
        