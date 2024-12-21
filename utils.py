import json
import re
import time
from typing import Optional, Union, List, Dict
from models import Tool, ToolCall, ToolFunction, MessageContent

def is_last_message_tool(messages):
    """Check if the last message in the list has a 'tool' role"""
    if not messages:
        return False
    return messages[-1].role == "tool"

def extract_tool_execution_messages(messages):
    """
    Extract and format tool execution messages into a single response.
    
    Args:
        messages (list): List of conversation messages
        
    Returns:
        dict: Formatted message combining query and response
    """

    try:
        tool_output = messages[-1].content if messages[-1] else ''
        user_query = messages[-3].content if messages[-3] else ''
        user_prompt = """Given the question and answer, provide a natural and helpful response that:
1. Answers the user's original question
2. Uses clear, concise language
3. Do not add any new information"""

        formatted_message = {
            'role': 'user',
            'content': f"{user_prompt}\nQuestion: {user_query}\nAnswer: {tool_output}"
        }
        
        return [formatted_message]  # Return as single-element list
        
    except (AttributeError, IndexError):
        return messages

def create_system_message() -> List[Dict[str, str]]:
    """Create the system message for the conversation
    
    Returns:
        List[Dict[str, str]]: List containing system role message
    """
    return [{
        "role": "system",
        "content": """You are a helpful assistant with access to the following function calls. 
                     Your task is to produce a sequence of function calls necessary to generate response to the user utterance. 
                     Use the following function calls as required."""
    }]

def format_tools(tools):
    """Converts tool objects with nested function attributes into required format"""
    formatted_tools = []
    
    for tool in tools:
        if hasattr(tool, 'function'):
            formatted_tool = {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters
            }
            formatted_tools.append(formatted_tool)
    
    return formatted_tools

def prepare_user_messages(user_messages):
    """Process and format user messages"""
    formatted_messages = []
    for msg in user_messages:
        content = process_content(msg.content)
        formatted_messages.append({
            "role": msg.role,
            "content": content,
            "name": msg.name
        })
    return formatted_messages

def process_content(content: Optional[Union[str, List[MessageContent], List[Dict[str, str]]]] = None) -> str:
    """Process content to return a string regardless of input type"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if not content:
            return ""
        if isinstance(content[0], MessageContent):
            return content[0].text
        if isinstance(content[0], dict) and "text" in content[0]:
            return content[0]["text"]
    return ""

def clean_response(text):
    # Extract everything after the assistant role marker
    match = re.search(r'<\|start_of_role\|>assistant<\|end_of_role\|>(.*?)(?:<\|end_of_text\|>|$)', text, re.DOTALL)
    if not match:
        return text.strip(), None
    
    response = match.group(1).strip()
    print(f"Extracted response: {response}")
    
    # Check for tool calls in the response
    tool_calls = None
    try:
        # Additional regex for <|tool_call|> followed by JSON array
        func_match = re.search(r'(?<=<\|tool_call\|>)(\[.*\])', response, re.DOTALL)
        if not func_match:
            # Look for JSON structure            
            func_match = re.search(r'\{[\s\S]*\}', response, re.DOTALL)
        
        if func_match:
            # Assuming func_match.group(0) contains the JSON object as a string
            json_str = func_match.group(0)
            # Parse the JSON string
            json_obj = json.loads(json_str)
            # Extract the function name and arguments
            if isinstance(json_obj, list):
                tool_calls = []
                for obj in json_obj:
                    func_name = obj.get("name")
                    arguments = obj.get("arguments")
                    tool_calls.append(ToolCall(
                        id=f"call_{int(time.time())}",
                        type="function",
                        function=ToolFunction(
                            name=func_name,
                            arguments=json.dumps(arguments)
                        )
                    ))
            else:
                func_name = json_obj.get("name")
                arguments = json_obj.get("arguments")
                tool_calls = [ToolCall(
                    id=f"call_{int(time.time())}",
                    type="function",
                    function=ToolFunction(
                        name=func_name,
                        arguments=json.dumps(arguments)
                    )
                )]
            response = None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        
    return response, tool_calls