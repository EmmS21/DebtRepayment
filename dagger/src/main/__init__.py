from dagger import dag, function, object_type, Secret
import logging
import json
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@object_type
class DebtRepayment:
    @function
    async def fetch_data(self, apiKey: Secret, sheet: Secret, sheet_two: Secret, open_key: Secret, fred_str: Secret, name: str) -> str:
        """Returns a container that echoes whatever string argument is provided"""
        fetch_balance = await dag.fetch_spreadsheet_data().fetch_data(apiKey, sheet, 'Edited')
        logger.info(f"balance_resp: {fetch_balance}")
        fetch_balance = self.convert_amounts(fetch_balance)

        fetch_debt = await dag.fetch_spreadsheet_data().fetch_data(apiKey, sheet_two, name)
        fetch_debt = self.restructure_data(fetch_debt)
        sectors_of_interest = "Health Care,Information Technology,Financials,Energy"
        # stocks_data = await self.get_stocks(sectors_of_interest)
    
        decision = await self.run_agent(open_key, fred_str, fetch_debt)
        
        output = decision.get("decision", {}).get("output", "")
        input = decision.get("decision", {}).get("input", "")

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
        required_columns = ['Interest Checking - Fixed (XXX)', 'Savings Account (XXX)']
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
    
    async def run_agent(self, open_key, fred_str, fetch_debt) -> str:
            @tool  
            async def fetch_stocks(sectors_of_interest:str) -> str:
                """Fetches stock data for given sectors."""
                stocks_data = await dag.get_stocks().stocks(sectors_of_interest)
                return stocks_data
            
            @tool
            async def calculate_time_value(period: int, amount: str, rate: str) -> str:
                """Calculate the future value of an amount of money over a period with a given rate."""
                return dag.calculate_time_value().calculate(period, amount, rate, fred_str)
            
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
                        ("system", """You are an autonomous financial agent. Your task is to analyze the given stock and debt data and suggest an optimal payment strategy assuming I have a balance of $2,000.

                Follow these steps:

                1. **Question**: What is the optimal strategy for allocating $2,000 between debt repayment and investments?
                2. **Thought**: I need to analyze the provided debt details and fetch stock performance data.
                3. **Observation**: Review the provided debt details, including amounts, minimum payments, and interest rates.

                4. **Tool/Action**: Retrieve stock performance data.
                5. **Tool/Action Input**: Use the specified sectors to fetch stock data, including average returns and current prices of top stocks.
                6. **Observation**: Record the average returns and current prices of the top stocks from Health Care and Information Technology sectors.

                7. **Thought**: Decide on a reasonable time period for investment calculations based on how long it would likely take to pay back the given debt.
                8. **Tool/Action**: Calculate future investment values.
                9. **Tool/Action Input**: Use the chosen time period and investment amounts to calculate future values.
                10. **Observation**: Record the future values of the investments.

                11. **Thought**: Compare the potential investment returns with the benefits of debt repayment.
                12. **Tool/Action**: Calculate the benefits of debt repayment over the same period.
                13. **Tool/Action Input**: Use the debt details and minimum payments for calculations.
                14. **Observation**: Record the benefits of debt repayment.

                15. **Thought**: Determine the optimal allocation of $2,000 between investments and debt repayment.
                16. **Observation**: Provide a detailed analysis with specific recommendations.

                Final Answer: Return a JSON with:
                1. Each debt and the amount to be paid off this month.
                2. Stocks (if applicable) and the amount to invest + the return over the chosen period.
                3. A concise rationale explaining the decision and quantifiable benefits.                
                """),
                        MessagesPlaceholder("chat_history", optional=True),
                        ("human", "{input}"),
                        MessagesPlaceholder("agent_scratchpad")
                    ]
                )
                toolkit = [fetch_stocks, calculate_time_value]
                agent = create_openai_tools_agent(llm, toolkit, simple_prompt_template)
                agent_executor = AgentExecutor(
                                               agent=agent, 
                                               tools=toolkit, 
                                               verbose=True,
                                               handle_parsing_errors=True,
                                               max_iterations=30,
                                               max_execution_time=180
                                )

                logger.info("AgentExecutor initialized")
            except Exception as e:
                logger.error(f"Error initializing AgentExecutor: {e}")
                raise e

            try:
                result = await agent_executor.ainvoke(input_data)

                # Assuming 'result' contains the final decision and intermediate steps
                logger.info(f"Final decision: {result}")

                return {"decision": result, "agent_scratchpad": input_data["agent_scratchpad"]}


            except Exception as e:
                logger.error(f"Error running agent: {e}")
                raise e
        
