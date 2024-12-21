using OpenAI;
using OpenAI.Assistants;
using OpenAI.Chat;
using System;
using System.ClientModel;
using System.ClientModel.Primitives;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace IBMGranite
{
    public class FunctionExamples
    {
        public static void Run()
        {
            // Initialize the OpenAI API client and test simple chat
            var openAIOptions = new OpenAIClientOptions
            {
                Endpoint = new Uri("http://127.0.0.1:8000/v1"),
                RetryPolicy = new ClientRetryPolicy(0),
                NetworkTimeout = TimeSpan.FromMinutes(5) // Increase the timeout duration to 5 minutes
            };
            var apiKey = new ApiKeyCredential("FAKE_API");
            var model = "granite-3.0-2b-instruct";
            var client = new ChatClient(model, apiKey, openAIOptions);
            
            List<ChatMessage> messages =
            [
                new UserChatMessage("What's the weather like today in Los Angeles, CA?"),
            ];
            ChatCompletionOptions options = new()
            {
                Tools = { GetCurrentWeatherTool },
                MaxOutputTokenCount = 2048,
                Temperature = 1
            };

            bool requiresAction;

            do
            {
                requiresAction = false;
                ChatCompletion completion = client.CompleteChat(messages, options);

                switch (completion.FinishReason)
                {
                    case ChatFinishReason.Stop:
                        {
                            // Add the assistant message to the conversation history.
                            messages.Add(new AssistantChatMessage(completion));
                            break;
                        }

                    case ChatFinishReason.ToolCalls:
                        {
                            // First, add the assistant message with tool calls to the conversation history.
                            messages.Add(new AssistantChatMessage(completion));

                            // Then, add a new tool message for each tool call that is resolved.
                            foreach (ChatToolCall toolCall in completion.ToolCalls)
                            {
                                switch (toolCall.FunctionName)
                                {
                                    case nameof(GetCurrentWeather):
                                        {
                                            // The arguments that the model wants to use to call the function are specified as a
                                            // stringified JSON object based on the schema defined in the tool definition. Note that
                                            // the model may hallucinate arguments too. Consequently, it is important to do the
                                            // appropriate parsing and validation before calling the function.
                                            using JsonDocument argumentsJson = JsonDocument.Parse(toolCall.FunctionArguments);
                                            bool hasLocation = argumentsJson.RootElement.TryGetProperty("location", out JsonElement location);
                                            
                                            if (!hasLocation)
                                            {
                                                throw new ArgumentNullException(nameof(location), "The location argument is required.");
                                            }

                                            string toolResult = GetCurrentWeather(location.GetString());
                                            messages.Add(new ToolChatMessage(toolCall.Id, toolResult));
                                            break;
                                        }

                                    default:
                                        {
                                            // Handle other unexpected calls.
                                            throw new NotImplementedException();
                                        }
                                }
                            }

                            requiresAction = true;
                            break;
                        }

                    case ChatFinishReason.Length:
                        throw new NotImplementedException("Incomplete model output due to MaxTokens parameter or token limit exceeded.");

                    case ChatFinishReason.ContentFilter:
                        throw new NotImplementedException("Omitted content due to a content filter flag.");

                    case ChatFinishReason.FunctionCall:
                        throw new NotImplementedException("Deprecated in favor of tool calls.");

                    default:
                        throw new NotImplementedException(completion.FinishReason.ToString());
                }
            } while (requiresAction);

            foreach (ChatMessage message in messages)
            {
                switch (message)
                {
                    case UserChatMessage userMessage:
                        Console.WriteLine($"[USER]:");
                        Console.WriteLine($"{userMessage.Content[0].Text}");
                        Console.WriteLine();
                        break;

                    case AssistantChatMessage assistantMessage when assistantMessage.Content.Count > 0:
                        Console.WriteLine($"[ASSISTANT]:");
                        Console.WriteLine($"{assistantMessage.Content[0].Text}");
                        Console.WriteLine();
                        break;

                    case ToolChatMessage:
                        // Do not print any tool messages; let the assistant summarize the tool results instead.
                        break;

                    default:
                        break;
                }
            }
        }

        private static string GetCurrentWeather(string location)
        {
            // Call the weather API here.
            return "65 fahrenheit";
        }

        private static readonly ChatTool GetCurrentWeatherTool = ChatTool.CreateFunctionTool(
            functionName: nameof(GetCurrentWeather),
            functionDescription: "Get the current weather in a given user location",
            functionParameters: BinaryData.FromBytes("""
                {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. Boston, MA"
                        }
                    },
                    "required": [ "location" ]
                }
                """u8.ToArray())
        );
    }
}
