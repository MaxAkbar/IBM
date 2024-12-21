from request_models import ChatCompletionRequest
from response_models import ChatCompletionResponse, Choice, Usage
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import clean_response, prepare_user_messages, is_last_message_tool, extract_tool_execution_messages, create_system_message, format_tools
import torch
import uvicorn
import time
import json

# Global variables for model and tokenizer
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    try:
        print("Loading model and tokenizer...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model_path = "ibm-granite/granite-3.1-8b-instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device if device == "cuda" else None
        )
        model.eval()
        print("Model and tokenizer loaded successfully!")
        yield
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    finally:
        print("Shutting down application...")

app = FastAPI(
    title="IBM Granite Chat API (OpenAI Compatible)",
    description="A REST API service for IBM Granite LLM with OpenAI-compatible endpoints",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        print(json.dumps(request.model_dump(), indent=4))
        print(f"Received request with tools: {bool(request.tools)}")
        
        device = next(model.parameters()).device
        
        # Separate tool and user message handling
        # Check if last message is a tool call
        has_tool_last = is_last_message_tool(request.messages)
        system_message = []
        tool_messages = []
        
        if has_tool_last:
            print(f"Last message is tool: {has_tool_last}")

            # now take the last and the third message from last and generate the response
            user_messages = extract_tool_execution_messages(request.messages)
        else:
            tool_messages = format_tools(request.tools)
            # if we have tools then get the system message
            if tool_messages:
                system_message = create_system_message()
            
            user_messages = prepare_user_messages(request.messages)
        
        # Combine messages in correct order
        messages = system_message + user_messages
        
        print(f"Prepared messages: {messages}")
        
        # Format chat
        if not tool_messages:
            chat = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            chat = tokenizer.apply_chat_template(
                messages,
                tools= tool_messages,
                tokenize=False,
                add_generation_prompt=True
            )

        print(f"Formatted chat: {chat}")
        # Generate response
        print("Generating response...")

        input_tokens = tokenizer(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            output_tokens = model.generate(
                **input_tokens,
                max_new_tokens=request.max_completion_tokens or 100,
                temperature=request.temperature or 1.0,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Process the response
        output_text = tokenizer.batch_decode(output_tokens)[0]
        clean_text, tool_calls = clean_response(output_text)
        
        # Count tokens
        prompt_tokens = len(input_tokens.input_ids[0])
        completion_tokens = len(output_tokens[0]) - len(input_tokens.input_ids[0])
        
        # Prepare the message
        message_dict = {
            "role": "assistant",
        }
        
        if tool_calls:
            message_dict["tool_calls"] = [tool_call.model_dump() for tool_call in tool_calls]
            message_dict["content"] = None
        else:
            message_dict["content"] = clean_text
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=message_dict,
                    finish_reason="tool_calls" if tool_calls else "stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        print(json.dumps(response.model_dump(), indent=4))

        return response
    
    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "data": [
            {
                "id": "granite-3.0-8b-instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ibm",
                "permission": [],
                "root": "granite-3.0-8b-instruct",
                "parent": None
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Check if the service is healthy and model is loaded"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the application with Uvicorn:
    # uvicorn chat_service:app --host 127.0.0.1 --port 8000 --reload
    uvicorn.run(app, host="127.0.0.1", port=8000)