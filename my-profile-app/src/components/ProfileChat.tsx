"use client";

// src/components/ProfileChat.js (or place directly in pages/index.js)
import { type FC, useState, useRef, useEffect, useCallback } from 'react';
import { Send, Mic, Square } from 'lucide-react'; // Using lucide-react for icons
import { fetchAuthSession } from 'aws-amplify/auth'; // Import the function

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
  // **NEW**: State to track if Speech Recognition is supported and initialized
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
  }, []); // No external dependencies, safe to define early

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
  }, []); // No external dependencies, safe to define early


  // Handles sending the message (from input or voice)
  // Now defined before the speech recognition useEffect
  const handleSendMessage = useCallback(async (messageText: string | null = null) => {
    const textToSend = (messageText ?? userInput).trim();
    if (!textToSend || isThinking) return;

    if (typeof window !== 'undefined' && window.speechSynthesis && window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
    }

    let idToken = '';
    try {
        const session = await fetchAuthSession(); // Get current session
        idToken = session.tokens?.idToken?.toString() ?? ''; // Get the ID token string
        if (!idToken) {
            console.error("No ID token found in session. User might not be logged in.");
            // Handle appropriately - maybe force sign-in or show error
            addMessage("Error: Authentication token not found. Please sign in again.", 'system');
            setIsThinking(false); // Ensure thinking state is reset
            return; // Stop before making the fetch call
        }
    } catch (error) {
        console.error("Error fetching auth session:", error);
        addMessage("Error: Could not retrieve authentication session.", 'system');
        setIsThinking(false);
        return;
    }

    addMessage(textToSend, 'user');
    // **FIXED**: Always clear the input after adding the user message, regardless of source
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
          'Authorization': `Bearer ${idToken}` // Use the ID token for authentication
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
        } catch (e) { console.warn("Could not parse error response body as JSON."); }
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
    // Dependencies for handleSendMessage:
  }, [userInput, addMessage, backendUrl, isThinking, speak]); // Added 'speak' as dependency

  // --- Speech Recognition Setup (useEffect runs once on mount) ---
  useEffect(() => {
    // Check if running on the client-side
    if (typeof window === 'undefined') {
        console.log("Not running in browser, skipping Speech Recognition setup.");
        return;
    }

    let recognition: SpeechRecognition | null = null;
    try {
      const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognitionAPI) {
        setStatusMessage('Speech recognition not supported in this browser.');
        console.warn('Speech Recognition API not supported.');
        setIsSpeechRecognitionSupported(false); // Explicitly set state if not supported
        return;
      }

      // **NEW**: Set state to true indicating support
      setIsSpeechRecognitionSupported(true);

      recognition = new SpeechRecognitionAPI();
      recognition.continuous = false;
      recognition.lang = 'en-US';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognitionRef.current = recognition; // Still store the instance in the ref

      // --- Event Handlers ---
      recognition.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        console.log('Transcript received:', transcript);
        if (transcript) {
          handleSendMessage(transcript);
        }
        setIsListening(false);
      };

      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        if (event.error === 'aborted') {
          console.log('Speech recognition aborted (potentially expected).');
          if (isListening) { setIsListening(false); setStatusMessage(''); }
          return;
        }
        console.error('Speech recognition error:', event.error, event.message);
        let errorMsg = `Speech error: ${event.error}`;
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
          errorMsg = 'Mic permission denied. Please allow microphone access in browser settings.';
        } else if (event.error === 'no-speech') {
          errorMsg = 'No speech detected. Please try again.';
        } else if (event.error === 'audio-capture') {
            errorMsg = 'Microphone not found or not working.';
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
        console.log('Speech recognition service disconnected.');
        setIsListening(false);
        if (statusMessage === 'Listening...' || statusMessage === 'Speech detected...') {
             setStatusMessage('');
        }
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
      setIsSpeechRecognitionSupported(false); // Ensure state is false on init error
    }

    // Cleanup function
    return () => {
      try {
        const currentRecognition = recognitionRef.current;
        if (currentRecognition) {
          console.log('Cleaning up speech recognition.');
          currentRecognition.onresult = null;
          currentRecognition.onerror = null;
          currentRecognition.onend = null;
          currentRecognition.onaudiostart = null;
          currentRecognition.onspeechstart = null;
          currentRecognition.onspeechend = null;
          currentRecognition.abort();
          recognitionRef.current = null;
        }
      } catch (error) {
        console.warn('Error during speech recognition cleanup:', error);
      }
    };
    // handleSendMessage is now defined above, so this dependency is safe.
  }, [handleSendMessage]); // Dependency array remains correct

  // --- Auto-scroll messages ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // --- More Core Functions (can be defined after effects if not dependencies) ---

  // Starts the speech recognition process
  const startListening = useCallback(() => {
    const currentRecognition = recognitionRef.current;
    // Also check if supported before trying to start
    if (!isSpeechRecognitionSupported || !currentRecognition || isListening || isThinking) {
        console.log("Cannot start listening: Not supported, not ready, already listening, or assistant is thinking.");
        if (!isSpeechRecognitionSupported) setStatusMessage("Speech recognition not supported.");
        return;
    }
    try {
      setUserInput(''); // Clear input field when starting mic
      setStatusMessage('Initializing microphone...');
      currentRecognition.start();
    } catch (error) {
      console.error("Error starting recognition:", error);
      if (error instanceof DOMException && error.name === 'InvalidStateError') {
          setStatusMessage('Recognition is already active.');
          setIsListening(true);
      } else {
          setStatusMessage('Mic error. Please ensure permission is granted and mic is working.');
          setIsListening(false);
      }
    }
  }, [isListening, isThinking, isSpeechRecognitionSupported]); // Added isSpeechRecognitionSupported dependency

  // Stops the speech recognition process
  const stopListening = useCallback(() => {
    const currentRecognition = recognitionRef.current;
    if (!currentRecognition || !isListening) {
        console.log("Cannot stop listening: Recognition not active or not available.");
        return;
    }
    try {
      console.log('Manually stopping recognition.');
      currentRecognition.abort();
    } catch (error) {
      console.error("Error stopping recognition:", error);
      setStatusMessage('Error stopping microphone');
      setIsListening(false);
    }
  }, [isListening]);

  // Handle Enter key press in input field
  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !isThinking) {
      event.preventDefault();
      handleSendMessage(); // Call the function defined above
    }
  };

  // --- JSX Rendering ---
  return (
    // Added dark mode support via class
    <div className="bg-gradient-to-br from-gray-100 via-white to-gray-200 dark:from-gray-800 dark:via-gray-900 dark:to-black flex items-center justify-center min-h-screen p-4 font-sans">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-2xl mx-auto overflow-hidden flex flex-col" style={{ height: 'calc(100vh - 4rem)', maxHeight: '800px' }}>
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 dark:from-blue-700 dark:to-indigo-800 text-white p-4 shadow-md flex-shrink-0">
          <h1 className="text-xl font-semibold text-center">Ask Srimanth's AI Assistant</h1>
        </div>

        {/* Chat Messages Area */}
        <div id="messages" className="flex-grow overflow-y-auto p-4 md:p-6 space-y-4 bg-gray-50 dark:bg-gray-900 scrollbar-thin scrollbar-thumb-blue-500 scrollbar-track-gray-200 dark:scrollbar-thumb-blue-700 dark:scrollbar-track-gray-700">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`p-3 rounded-lg max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg shadow-md break-words text-sm md:text-base ${
                  msg.sender === 'user'
                    ? 'bg-blue-500 dark:bg-blue-600 text-white rounded-br-none'
                    : msg.sender === 'assistant'
                    ? 'bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-100 rounded-bl-none'
                    : 'bg-red-100 dark:bg-red-900 border border-red-300 dark:border-red-700 text-red-700 dark:text-red-200 w-full text-center text-xs md:text-sm' // System/Error message style
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {/* Thinking Indicator */}
          {isThinking && (
            <div className="flex justify-start">
              <div className="p-3 rounded-lg shadow-md bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-100 rounded-bl-none animate-pulse text-sm md:text-base">
                <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mr-1 animate-bounce" style={{animationDelay: '0ms'}}></span>
                <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mr-1 animate-bounce" style={{animationDelay: '150ms'}}></span>
                <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></span>
              </div>
            </div>
          )}
          {/* Empty div to ensure scrolling to the bottom works */}
          <div ref={messagesEndRef} className="h-1" />
        </div>

        {/* Status Bar */}
        <div className="px-4 py-1 text-xs md:text-sm text-center text-gray-500 dark:text-gray-400 h-6 flex-shrink-0">
          {statusMessage}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 dark:border-gray-700 p-3 md:p-4 bg-white dark:bg-gray-800 flex-shrink-0">
          <div className="flex items-center space-x-2">
            <input
              type="text"
              id="user-input"
              aria-label="Chat input"
              placeholder={isListening ? "Listening..." : "Type your question or use the mic..."}
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyDown={handleKeyPress} // Use onKeyDown for better Enter key handling consistency
              disabled={isListening || isThinking} // Disable input while listening or thinking
              className="flex-grow p-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-600 transition duration-150 disabled:opacity-70 disabled:bg-gray-100 dark:disabled:bg-gray-700 dark:bg-gray-700 dark:text-gray-100 dark:placeholder-gray-400"
            />
            {/* Send Button */}
            <button
              onClick={() => handleSendMessage()}
              title="Send Message"
              aria-label="Send Message"
              disabled={!userInput.trim() || isListening || isThinking}
              className="bg-blue-500 hover:bg-blue-600 dark:bg-blue-600 dark:hover:bg-blue-700 text-white p-3 rounded-lg transition duration-150 shadow focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-600 focus:ring-offset-1 dark:focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send size={20} />
            </button>
            {/* Microphone Button - **FIXED**: Render based on state variable */}
            {isSpeechRecognitionSupported && ( // Only render mic button if supported (based on state)
                <button
                onClick={isListening ? stopListening : startListening}
                title={isListening ? "Stop Listening" : "Ask with Voice"}
                aria-label={isListening ? "Stop Listening" : "Ask with Voice"}
                disabled={isThinking} // Only disable if assistant is thinking
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
