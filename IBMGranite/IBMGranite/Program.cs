using IBMGranite;
using OpenAI;
using OpenAI.Chat;
using OpenAI.Embeddings;
using System.ClientModel;

//// Initialize the OpenAI API client and test simple chat
//var openAIOptions = new OpenAIClientOptions
//{
//    Endpoint = new Uri("http://127.0.0.1:8000/v1")
//};
//var apiKey = new ApiKeyCredential("FAKE_API");
//var model = "granite-3.0-2b-instruct";
//var chatClient = new ChatClient(model, apiKey, openAIOptions);
//var inputText = "Hello, what can you do?";
//var chatMessage = ChatMessage.CreateUserMessage(inputText);

//var chatCompletion = chatClient.CompleteChat(chatMessage);
//var response = chatCompletion.Value.Content[0].Text;

//// Output the embedding vector
//Console.WriteLine("Chat Response: ");
//Console.WriteLine(response);

// Now test the function call
FunctionExamples.Run();