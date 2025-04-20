"use client";

// src/components/ProfileChat.js (or place directly in pages/index.js)
import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Mic, Square } from 'lucide-react'; // Using lucide-react for icons
import { fetchAuthSession } from 'aws-amplify/auth';

// Define the structure for a chat message
interface Message {
  id: number;
  text: string;
  sender: 'user' | 'assistant' | 'system'; // System for errors/status
}

// Define error type
interface FetchError extends Error {
  status?: number;
  statusText?: string;
}

// Extend Window interface for SpeechRecognition APIs
declare global {
    interface Window {
        SpeechRecognition?: typeof SpeechRecognition;
        webkitSpeechRecognition?: typeof SpeechRecognition;
        SpeechRecognitionEvent?: typeof SpeechRecognitionEvent;
        SpeechRecognitionErrorEvent?: typeof SpeechRecognitionErrorEvent;
    }
}

// The main chat component
export default function ProfileChat() {
  // State variables
  const [userInput, setUserInput] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([
    { id: Date.now(), text: "Hi there! I'm an AI assistant representing Srimanth. Feel free to ask about their skills and experience.", sender: 'assistant' },
  ]);
  const [isListening, setIsListening] = useState<boolean>(false);
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [isSpeechRecognitionSupported, setIsSpeechRecognitionSupported] = useState<boolean>(false);

  // Refs
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // --- Configuration ---
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000/query';

  // --- Core Functions (Defined before useEffect hooks that depend on them) ---

  // Adds a new message to the chat display
  const addMessage = useCallback((text: string, sender: 'user' | 'assistant' | 'system') => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { id: Date.now() + Math.random(), text, sender },
    ]);
  }, []);

  // Uses Text-to-Speech to speak the assistant's response
  const speak = useCallback((text: string) => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window && text) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'en-US';
      utterance.rate = 1;
      utterance.pitch = 1;
      utterance.onstart = () => console.log("Speech synthesis started.");
      utterance.onend = () => console.log("Speech synthesis finished.");
      utterance.onerror = (event) => console.error('Speech synthesis error:', event.error, event);
      window.speechSynthesis.speak(utterance);
    } else if (typeof window !== 'undefined' && !('speechSynthesis' in window)) {
      console.warn('Speech synthesis not supported in this browser.');
    }
  }, []);

  // Handles sending the message (from input or voice)
  const handleSendMessage = useCallback(async (messageText: string | null = null) => {
    const textToSend = (messageText ?? userInput).trim();
    if (!textToSend || isThinking) return;

    if (typeof window !== 'undefined' && window.speechSynthesis && window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
    }

    let idToken = '';
    try {
        const session = await fetchAuthSession();
        idToken = session.tokens?.idToken?.toString() ?? '';
        if (!idToken) {
            console.error("No ID token found in session. User might not be logged in.");
            addMessage("Error: Authentication token not found. Please sign in again.", 'system');
            setIsThinking(false);
            return;
        }
    } catch (error) {
        console.error("Error fetching auth session:", error);
        addMessage("Error: Could not retrieve authentication session.", 'system');
        setIsThinking(false);
        return;
    }

    addMessage(textToSend, 'user');
    setUserInput('');

    setIsThinking(true);
    setStatusMessage('Assistant is thinking...');

    try {
      console.log(`Sending to backend: ${backendUrl}`, { query: textToSend });
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Authorization': `Bearer ${idToken}`
        },
        body: JSON.stringify({ query: textToSend }),
        signal: AbortSignal.timeout(30000)
      });

      setIsThinking(false);
      setStatusMessage('');

      if (!response.ok) {
        let errorDetails = `Error: ${response.status} ${response.statusText}`;
        try {
          const errorData = await response.json();
          errorDetails = `Error: ${errorData.error || JSON.stringify(errorData)}`;
        } catch {
          console.warn("Could not parse error response body as JSON.");
        }
        const fetchError = new Error(errorDetails) as FetchError;
        fetchError.status = response.status;
        fetchError.statusText = response.statusText;
        throw fetchError;
      }

      const data = await response.json();
      const assistantResponse = data.answer || "Sorry, I received an empty response.";
      addMessage(assistantResponse, 'assistant');
      speak(assistantResponse);

    } catch (error: unknown) {
      setIsThinking(false);
      setStatusMessage('');
      let errorMessage = 'An unknown error occurred.';
      if (error instanceof Error) {
          errorMessage = error.name === 'AbortError' ? 'Request timed out. Please try again.' : (error.message || 'Could not reach the backend.');
      }
      console.error('Error sending message:', error);
      addMessage(`Error: ${errorMessage}`, 'system');
      setStatusMessage(`Error: ${errorMessage}`);
    }
  }, [userInput, addMessage, backendUrl, isThinking, speak]);

  // --- Speech Recognition Setup ---
  useEffect(() => {
    if (typeof window === 'undefined') {
        console.log("Not running in browser, skipping Speech Recognition setup.");
        return;
    }

    let recognition: SpeechRecognition | null = null;
    
    const initializeRecognition = () => {
      const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognitionAPI) {
        setStatusMessage('Speech recognition not supported in this browser.');
        console.warn('Speech Recognition API not supported.');
        setIsSpeechRecognitionSupported(false);
        return null;
      }

      const recognition = new SpeechRecognitionAPI();
      recognition.continuous = false;
      recognition.lang = 'en-US';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      return recognition;
    };

    try {
      recognition = initializeRecognition();
      if (!recognition) return;

      setIsSpeechRecognitionSupported(true);
      recognitionRef.current = recognition;

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        console.log('Transcript received:', transcript);
        if (transcript) {
          handleSendMessage(transcript);
        }
        setIsListening(false);
        setStatusMessage('');
      };

      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        if (event.error === 'aborted') {
          console.log('Speech recognition aborted.');
          setIsListening(false);
          setStatusMessage('');
          return;
        }
        
        console.error('Speech recognition error:', event.error);
        let errorMsg = 'Speech error occurred';
        
        switch (event.error) {
          case 'not-allowed':
          case 'service-not-allowed':
            errorMsg = 'Mic permission denied. Please allow microphone access.';
            break;
          case 'no-speech':
            errorMsg = 'No speech detected. Please try again.';
            break;
          case 'audio-capture':
            errorMsg = 'Microphone not found or not working.';
            break;
          default:
            errorMsg = `Speech error: ${event.error}`;
        }
        
        setStatusMessage(errorMsg);
        setIsListening(false);
      };

      recognition.onaudiostart = () => {
        console.log('Audio capture started.');
        setStatusMessage('Listening...');
        setIsListening(true);
      };

      recognition.onend = () => {
        console.log('Speech recognition ended.');
        if (statusMessage === 'Listening...' || statusMessage === 'Speech detected...') {
          setStatusMessage('');
        }
        setIsListening(false);
      };

      recognition.onspeechstart = () => {
        console.log('Speech detected.');
        setStatusMessage('Speech detected...');
      };

      recognition.onspeechend = () => {
        console.log('Speech ended.');
      };

    } catch (error) {
      console.error('Error initializing speech recognition:', error);
      setStatusMessage('Error initializing speech recognition');
      setIsListening(false);
      setIsSpeechRecognitionSupported(false);
    }

    return () => {
      if (recognitionRef.current) {
        try {
          if (isListening) {
            recognitionRef.current.abort();
          }
          recognitionRef.current = null;
        } catch (error) {
          console.warn('Error during speech recognition cleanup:', error);
        }
      }
    };
  }, []); // Only run once on mount

  // Starts the speech recognition process
  const startListening = useCallback(() => {
    if (!isSpeechRecognitionSupported || !recognitionRef.current || isListening || isThinking) {
      console.log("Cannot start listening:", {
        supported: isSpeechRecognitionSupported,
        hasRecognition: !!recognitionRef.current,
        isListening,
        isThinking
      });
      if (!isSpeechRecognitionSupported) {
        setStatusMessage("Speech recognition not supported.");
      }
      return;
    }

    try {
      setUserInput('');
      // Start recognition first, then set status
      recognitionRef.current.start();
      setStatusMessage('Listening...');
    } catch (error) {
      console.error("Error starting recognition:", error);
      if (error instanceof DOMException && error.name === 'InvalidStateError') {
        console.log('Recognition was already active, resetting state...');
        setStatusMessage('');
        setIsListening(false);
      } else {
        setStatusMessage('Mic error. Please ensure permission is granted.');
        setIsListening(false);
      }
    }
  }, [isSpeechRecognitionSupported, isListening, isThinking]);

  // Stops the speech recognition process
  const stopListening = useCallback(() => {
    if (!recognitionRef.current) {
      console.log("No recognition instance available.");
      return;
    }

    try {
      console.log('Stopping recognition...');
      recognitionRef.current.abort();
      setStatusMessage('');
      setIsListening(false);
    } catch (error) {
      console.error("Error stopping recognition:", error);
      setStatusMessage('Error stopping microphone');
      setIsListening(false);
    }
  }, []);

  // Handle Enter key press in input field
  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !isThinking) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  // --- JSX Rendering ---
  return (
    <div className="bg-gradient-to-br from-gray-100 via-white to-gray-200 dark:from-gray-800 dark:via-gray-900 dark:to-black flex items-center justify-center min-h-screen p-4 font-sans">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-2xl mx-auto overflow-hidden flex flex-col" style={{ height: 'calc(100vh - 4rem)', maxHeight: '800px' }}>
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 dark:from-blue-700 dark:to-indigo-800 text-white p-4 shadow-md flex-shrink-0">
          <h1 className="text-xl font-semibold text-center">Ask Srimanth's AI Assistant</h1>
        </div>

        <div id="messages" className="flex-grow overflow-y-auto p-4 md:p-6 space-y-4 bg-gray-50 dark:bg-gray-900 scrollbar-thin scrollbar-thumb-blue-500 scrollbar-track-gray-200 dark:scrollbar-thumb-blue-700 dark:scrollbar-track-gray-700">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`p-3 rounded-lg max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg shadow-md break-words text-sm md:text-base ${
                  msg.sender === 'user'
                    ? 'bg-blue-500 dark:bg-blue-600 text-white rounded-br-none'
                    : msg.sender === 'assistant'
                    ? 'bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-100 rounded-bl-none'
                    : 'bg-red-100 dark:bg-red-900 border border-red-300 dark:border-red-700 text-red-700 dark:text-red-200 w-full text-center text-xs md:text-sm'
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {isThinking && (
            <div className="flex justify-start">
              <div className="p-3 rounded-lg shadow-md bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-100 rounded-bl-none animate-pulse text-sm md:text-base">
                <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mr-1 animate-bounce" style={{animationDelay: '0ms'}} />
                <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mr-1 animate-bounce" style={{animationDelay: '150ms'}} />
                <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}} />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} className="h-1" />
        </div>

        <div className="px-4 py-1 text-xs md:text-sm text-center text-gray-500 dark:text-gray-400 h-6 flex-shrink-0">
          {statusMessage}
        </div>

        <div className="border-t border-gray-200 dark:border-gray-700 p-3 md:p-4 bg-white dark:bg-gray-800 flex-shrink-0">
          <div className="flex items-center space-x-2">
            <input
              type="text"
              id="user-input"
              aria-label="Chat input"
              placeholder={isListening ? "I&apos;m thinking..." : "Type your question or use the mic..."}
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyDown={handleKeyPress}
              disabled={isListening || isThinking}
              className="flex-grow p-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-600 transition duration-150 disabled:opacity-70 disabled:bg-gray-100 dark:disabled:bg-gray-700 dark:bg-gray-700 dark:text-gray-100 dark:placeholder-gray-400"
            />
            <button
              onClick={() => handleSendMessage()}
              title="Send Message"
              aria-label="Send Message"
              disabled={!userInput.trim() || isListening || isThinking}
              className="bg-blue-500 hover:bg-blue-600 dark:bg-blue-600 dark:hover:bg-blue-700 text-white p-3 rounded-lg transition duration-150 shadow focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-600 focus:ring-offset-1 dark:focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send size={20} />
            </button>
            {isSpeechRecognitionSupported && (
                <button
                onClick={isListening ? stopListening : startListening}
                title={isListening ? "Stop Listening" : "Ask with Voice"}
                aria-label={isListening ? "Stop Listening" : "Ask with Voice"}
                disabled={isThinking}
                className={`p-3 rounded-lg transition duration-150 shadow focus:outline-none focus:ring-2 focus:ring-offset-1 dark:focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed ${
                    isListening
                    ? 'bg-red-500 hover:bg-red-600 dark:bg-red-600 dark:hover:bg-red-700 text-white focus:ring-red-500 dark:focus:ring-red-600 animate-pulse'
                    : 'bg-indigo-500 hover:bg-indigo-600 dark:bg-indigo-600 dark:hover:bg-indigo-700 text-white focus:ring-indigo-500 dark:focus:ring-indigo-600'
                }`}
                >
                {isListening ? <Square size={20} /> : <Mic size={20} />}
                </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
