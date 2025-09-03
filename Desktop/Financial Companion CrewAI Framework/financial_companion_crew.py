# Financial Companion CrewAI Framework
# A multi-agent system for financial analysis, fraud detection, and product recommendations

import os
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool, CSVSearchTool
from langchain_google_genai import ChatGoogleGenerativeAI

# Data processing and ML imports
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pyod.models.iforest import IForest
import warnings
warnings.filterwarnings('ignore')

class FinancialCompanionCrew:
    """
    Financial Companion CrewAI Framework
    Processes customer transaction data, detects fraud, and recommends financial products
    """
    
    def __init__(self):
        """Initialize the crew with Google Gemini LLM"""
        # Load Google Gemini API key from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize Google Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_tokens=4000
        )
        
        # Initialize tools
        self.file_read_tool = FileReadTool()
        # self.csv_search_tool = CSVSearchTool()  # Disabled due to OpenAI API requirement
        
        # Create agents and tasks
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        self.crew = self._create_crew()
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create all specialized agents for the financial companion system"""
        
        agents = {}
        
        # Agent 1: Data Processing Specialist
        agents['data_processor'] = Agent(
            role='Transaction Data Validator and Processor',
            goal='Clean, validate, and normalize customer transaction data while ensuring data privacy and security compliance, achieving 99.9% data quality standards',
            backstory="""You are a meticulous data engineer with 10+ years in fintech data processing, 
                        specializing in transaction data normalization and privacy-first data handling. 
                        You are expert in detecting data anomalies and ensuring regulatory compliance from day one.
                        You understand the importance of data quality in financial systems and always prioritize security.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.file_read_tool],
            llm=self.llm
        )
        
        # Agent 2: Risk Guardian (Fraud Detection)
        agents['risk_guardian'] = Agent(
            role='Fraud Detection and Risk Assessment Analyst',
            goal='Identify suspicious transaction patterns and generate multi-level fraud alerts (low/medium/high risk) with 95% accuracy while minimizing false positives',
            backstory="""You are a former bank security analyst with expertise in real-time fraud detection systems. 
                        You combine traditional rule-based approaches with machine learning to catch evolving fraud patterns 
                        while protecting legitimate customers from unnecessary blocks. You understand the balance between 
                        security and customer experience.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Agent 3: Financial Health Advisor
        agents['health_advisor'] = Agent(
            role='Spending Pattern Analyst and Financial Wellness Expert',
            goal='Analyze 6-12 months of transaction history to identify spending patterns, calculate key financial health metrics, and generate actionable savings recommendations',
            backstory="""You are a certified financial planner who transitioned to fintech, specializing in 
                        micro-finance and financial inclusion. You are expert at translating complex financial data 
                        into simple, actionable insights for customers with varying financial literacy levels. 
                        You focus on practical advice that customers can immediately implement.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Agent 4: Product Matchmaker
        agents['product_matchmaker'] = Agent(
            role='Financial Product Recommendation Specialist',
            goal='Match customer financial profiles with optimal financial products (savings, micro-loans, insurance, investments) based on industry-standard eligibility criteria and risk assessments',
            backstory="""You are a former product manager at a leading digital bank with deep knowledge of 
                        financial product design and customer segmentation. You specialize in creating personalized 
                        product recommendations that drive both customer value and business growth. You understand 
                        regulatory requirements and eligibility criteria for various financial products.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Agent 5: Report Synthesizer
        agents['report_synthesizer'] = Agent(
            role='Financial Insights Report Generator',
            goal='Create comprehensive, easy-to-understand financial health reports in both JSON and human-readable formats that non-finance experts can easily comprehend',
            backstory="""You are a financial communications expert who bridges the gap between complex financial 
                        analysis and customer understanding. You are expert at creating reports that comply with 
                        regulatory requirements while remaining accessible to all education levels. You know how to 
                        present complex data in simple, actionable formats.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        return agents
    
    def _create_tasks(self) -> List[Task]:
        """Create all tasks for the financial companion workflow"""
        
        tasks = []
        
        # Task 1: Data Validation and Processing
        tasks.append(Task(
            description="""Load and validate the customer transaction data. Your responsibilities include:
            1. Read the transaction data file (CSV or JSON format)
            2. Validate required fields: transaction_id, date_time, amount, merchant_name, category, transaction_type, account_id, channel
            3. Check data quality: missing values, duplicates, invalid dates, negative amounts where inappropriate
            4. Normalize transaction categories and clean merchant names
            5. Ensure data privacy compliance (mask sensitive information if needed)
            6. Create a clean dataset ready for analysis
            7. Generate a data quality report with completeness scores
            
            Focus on data integrity and security. Flag any suspicious data patterns or quality issues.""",
            expected_output="""A comprehensive data validation report including:
            - Data quality score (percentage)
            - Number of transactions processed
            - Data completeness by field
            - Any data quality issues found and resolved
            - Summary of data cleaning actions taken
            - Confirmation that data meets security standards""",
            agent=self.agents['data_processor']
        ))
        
        # Task 2: Fraud Detection Analysis
        tasks.append(Task(
            description="""Analyze recent transaction patterns (last 30 days) to detect potential fraud. Your analysis should include:
            1. Identify unusual transaction patterns using statistical methods
            2. Check for impossible travel (transactions in different locations within impossible timeframes)
            3. Detect abnormal transaction amounts compared to customer's typical behavior
            4. Identify suspicious merchant patterns or new merchant interactions
            5. Check transaction velocity (too many transactions in short time periods)
            6. Generate multi-level risk alerts:
               - LOW: Unusual but not immediately concerning (new merchant, different location)
               - MEDIUM: Suspicious patterns requiring customer notification
               - HIGH: High probability fraud requiring immediate action
            7. Provide specific reasoning for each alert
            
            Balance fraud detection with customer experience - avoid false positives.""",
            expected_output="""Fraud risk assessment report containing:
            - Overall fraud risk score (Low/Medium/High)
            - Specific fraud alerts with risk levels and detailed explanations
            - Suspicious transaction details and patterns identified
            - Recommended actions for each risk level
            - Summary of normal vs suspicious transaction patterns
            - Confidence scores for each fraud alert""",
            agent=self.agents['risk_guardian']
        ))
        
        # Task 3: Financial Health Analysis
        tasks.append(Task(
            description="""Analyze the customer's financial health using 6-12 months of transaction history:
            1. Calculate spending patterns by category (groceries, utilities, entertainment, etc.)
            2. Determine monthly income vs expenses and cash flow trends
            3. Identify savings potential based on spending habits
            4. Calculate key financial ratios and metrics
            5. Detect seasonal spending patterns and trends
            6. Assess financial stability and identify areas for improvement
            7. Generate personalized savings opportunities
            8. Evaluate debt-to-income ratio if loan payments are present
            9. Provide financial wellness score
            
            Focus on actionable insights that help customers improve their financial health.""",
            expected_output="""Comprehensive financial health analysis including:
            - Financial wellness score (0-100)
            - Monthly spending breakdown by category
            - Income vs expense analysis with trends
            - Identified savings opportunities with potential amounts
            - Financial stability indicators
            - Personalized recommendations for financial improvement
            - Key financial ratios and benchmarks
            - Seasonal spending patterns and insights""",
            agent=self.agents['health_advisor']
        ))
        
        # Task 4: Product Recommendation
        tasks.append(Task(
            description="""Based on the financial health analysis and customer profile, recommend appropriate financial products:
            1. Analyze customer's financial profile and needs
            2. Match customer segments with appropriate products:
               - Savings accounts for customers with surplus cash
               - Micro-loans for customers with temporary cash flow gaps
               - Insurance products for risk protection needs
               - Investment products based on risk appetite and surplus funds
            3. Apply industry-standard eligibility criteria
            4. Consider customer's transaction behavior and financial stability
            5. Prioritize products by relevance and potential customer benefit
            6. Explain why each product is recommended
            7. Include terms, benefits, and eligibility requirements
            
            Focus on products that genuinely benefit the customer's financial situation.""",
            expected_output="""Prioritized financial product recommendations including:
            - Top 3-5 recommended products with priority ranking
            - Detailed explanation of why each product fits the customer
            - Eligibility requirements and likelihood of approval
            - Expected benefits and potential returns/savings
            - Product terms and conditions summary
            - Risk assessment for each recommended product
            - Next steps for product application/enrollment""",
            agent=self.agents['product_matchmaker']
        ))
        
        # Task 5: Report Generation
        tasks.append(Task(
            description="""Synthesize all previous analyses into comprehensive financial companion reports:
            1. Create an executive summary of key findings
            2. Compile fraud alerts and security recommendations
            3. Present financial health insights in easy-to-understand language
            4. Include product recommendations with clear explanations
            5. Generate actionable next steps for the customer
            6. Create both technical (JSON) and customer-friendly (narrative) formats
            7. Ensure all content is accessible to non-finance experts
            8. Include compliance disclaimers and privacy notices
            9. Structure information logically with clear sections
            
            Make complex financial analysis understandable for everyday customers.""",
            expected_output="""Complete financial companion report package containing:
            1. Executive Summary (key insights in 2-3 paragraphs)
            2. Security Status (fraud alerts and recommendations)
            3. Financial Health Dashboard (scores, trends, and insights)
            4. Personalized Recommendations (savings tips and product suggestions)
            5. Action Plan (specific next steps for customer)
            6. Technical JSON output for system integration
            7. Customer-friendly PDF/HTML formatted report
            All content should be clear, actionable, and compliant with regulations.""",
            agent=self.agents['report_synthesizer']
        ))
        
        return tasks
    
    def _create_crew(self) -> Crew:
        """Create and configure the CrewAI crew"""
        return Crew(
            agents=list(self.agents.values()),
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            max_rpm=10
        )
    
    def process_customer_data(self, transaction_file: str, customer_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process customer financial data and generate comprehensive analysis
        
        Args:
            transaction_file: Path to transaction data file (CSV or JSON)
            customer_profile: Optional customer profile dictionary
            
        Returns:
            Dictionary containing all analysis results and recommendations
        """
        
        # Prepare inputs for the crew
        inputs = {
            "transaction_file": transaction_file,
            "customer_profile": customer_profile or {},
            "analysis_date": datetime.now().isoformat(),
            "lookback_period_months": 12,
            "fraud_analysis_days": 30
        }
        
        # Execute the crew workflow
        print("🚀 Starting Financial Companion Analysis...")
        print(f"📁 Processing file: {transaction_file}")
        print(f"📅 Analysis date: {inputs['analysis_date']}")
        print("-" * 50)
        
        try:
            # Run the crew
            result = self.crew.kickoff(inputs=inputs)
            
            # Process and return results
            return {
                "status": "success",
                "analysis_date": inputs['analysis_date'],
                "transaction_file": transaction_file,
                "results": result,
                "crew_execution": "completed"
            }
            
        except Exception as e:
            print(f"❌ Error during crew execution: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e),
                "analysis_date": inputs['analysis_date']
            }

def main():
    """
    Main function to demonstrate the Financial Companion Crew usage
    """
    print("Financial Companion CrewAI Framework")
    print("=" * 50)
    
    # Initialize the crew
    financial_crew = FinancialCompanionCrew()
    
    # Example customer profile (optional)
    customer_profile = {
        "customer_id": "CUST_001",
        "age": 28,
        "gender": "F",
        "region": "Lagos",
        "income_range": "50000-100000",
        "financial_goals": ["savings", "investment"],
        "risk_appetite": "moderate"
    }
    
    # Example usage - replace with your actual transaction file
    transaction_file = "customer_transactions.csv"
    
    # Check if transaction file exists
    if not os.path.exists(transaction_file):
        print(f"⚠️  Transaction file '{transaction_file}' not found.")
        print("📝 Please ensure your transaction data file is available.")
        print("\nExpected CSV format should include columns:")
        print("transaction_id, date_time, amount, merchant_name, category, transaction_type, account_id, channel")
        return
    
    # Process customer data
    results = financial_crew.process_customer_data(
        transaction_file=transaction_file,
        customer_profile=customer_profile
    )
    
    # Display results
    if results["status"] == "success":
        print("✅ Financial Companion Analysis Completed Successfully!")
        print(f"📊 Results: {results['results']}")
    else:
        print(f"❌ Analysis Failed: {results['error_message']}")

if __name__ == "__main__":
    main()
