"""
Utility functions for testing and validating LLM models.
This module provides functions to check if models are working correctly
and can handle both regular and JSON responses.
"""

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

def check_model(llm_model):
    """Check if a model can handle basic queries correctly.
    
    Args:
        llm_model: The language model to test
        
    Returns:
        bool: True if model is working correctly, False otherwise
    """
    print("Checking the normal model by asking 'What is the capital of France?'")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the users query accurately."),
        ("human", "What is the capital of {country}?")
    ])
    
    output_parser = StrOutputParser()
    chain = prompt | llm_model | output_parser
    response = chain.invoke({'country': 'France'})
    print(f"The model response is: {response}")

    if 'capital' in response.lower():
        print("The model is working correctly!")
        return True
    else:
        print("The model is not working correctly!")
        return False




