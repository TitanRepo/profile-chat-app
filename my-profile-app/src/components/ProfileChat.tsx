"use client";

// src/components/ProfileChat.tsx (or similar)
import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Mic, Square, StopCircle } from 'lucide-react';
import { fetchAuthSession } from 'aws-amplify/auth';

// Define the structure for a chat message
interface Message {
  id: number;
  text: string;
  sender: 'user' | 'assistant' | 'system';
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
        // Add SpeechSynthesisUtteranceEvent if needed for more specific typing
        SpeechSynthesisUtteranceEvent?: typeof SpeechSynthesisUtteranceEvent;
    }
}

// The main chat component
export default function ProfileChat() {
  // State variables
  const [userInput, setUserInput] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([
    { id: Date.now(), text: "Hi there! I'm an AI assistant representing Srimanth. Feel free to ask about my skills and experience.", sender: 'assistant' },
  ]);
  const [isListening, setIsListening] = useState<boolean>(false);
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [isSpeechRecognitionSupported, setIsSpeechRecognitionSupported] = useState<boolean>(false);

  // Refs
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // --- Configuration ---
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000/query';

  // --- Core Functions ---

  const addMessage = useCallback((text: string, sender: 'user' | 'assistant' | 'system') => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { id: Date.now() + Math.random(), text, sender },
    ]);
  }, []);

  // Updated speak function to manage isSpeaking state and handle interruptions
  const speak = useCallback((text: string) => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window && text) {
      if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
      }
      setIsSpeaking(false);

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'en-US';
      utterance.rate = 1;
      utterance.pitch = 1;

      utterance.onstart = () => {
        console.log("Speech synthesis started.");
        setIsSpeaking(true);
      };

      utterance.onend = () => {
        console.log("Speech synthesis finished or cancelled.");
        setIsSpeaking(false);
      };

      // Handle 'interrupted' error gracefully
      utterance.onerror = (event: SpeechSynthesisEvent | Event) => {
        // Check if it's the specific SpeechSynthesisUtteranceEvent for error property
        let errorType = 'unknown';
        if ('error' in event && typeof event.error === 'string') {
             errorType = event.error;
        } else if (event instanceof ErrorEvent) { // Fallback for generic ErrorEvent
             errorType = event.message;
        }

        if (errorType === 'interrupted') {
            // Log expected interruptions calmly
            console.log(`Speech synthesis interrupted (error type: ${errorType}). This is usually expected.`);
        } else {
            // Log other errors as actual problems
            console.error(`Speech synthesis error: ${errorType}`, event);
        }
        setIsSpeaking(false); // Ensure speaking state is reset on any error/interruption
      };

      window.speechSynthesis.speak(utterance);
    } else if (typeof window !== 'undefined' && !('speechSynthesis' in window)) {
      console.warn('Speech synthesis not supported in this browser.');
    }
  }, []); // No dependencies needed

  const handleStopSpeaking = useCallback(() => {
      if (typeof window !== 'undefined' && window.speechSynthesis) {
          console.log("Manually stopping speech synthesis.");
          window.speechSynthesis.cancel();
          setIsSpeaking(false);
      }
  }, []);

  const handleSendMessage = useCallback(async (messageText: string | null = null) => {
    const textToSend = (messageText ?? userInput).trim();
    if (!textToSend || isThinking) return;

    handleStopSpeaking(); // Stop any ongoing speech

    let idToken = '';
    try { /* ... auth logic ... */
        const session = await fetchAuthSession();
        idToken = session.tokens?.idToken?.toString() ?? '';
        if (!idToken) { console.error("No ID token found."); addMessage("Error: Auth token missing.", 'system'); return; }
    } catch (error) { console.error("Auth error:", error); addMessage("Error: Auth session failed.", 'system'); return; }

    const currentMessagesForHistory = [...messages];
    addMessage(textToSend, 'user');
    setUserInput('');
    setIsThinking(true);
    setStatusMessage('Assistant is thinking...');

    const requestBody = { query: textToSend, history: currentMessagesForHistory.slice(-10).filter(m => m.sender !== 'system').map(m => ({ sender: m.sender, text: m.text })) };

    try { /* ... fetch logic ... */
        console.log(`Sending to backend: ${backendUrl}`, requestBody);
        const response = await fetch(backendUrl, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': `Bearer ${idToken}` }, body: JSON.stringify(requestBody), signal: AbortSignal.timeout(30000) });
        setIsThinking(false); setStatusMessage('');
        if (!response.ok) { let errorDetails = `Error: ${response.status} ${response.statusText}`; try { const errorData = await response.json(); errorDetails = `Error: ${errorData.error || JSON.stringify(errorData)}`; } catch (e) { console.warn("Could not parse error JSON.", e); } const fetchError = new Error(errorDetails) as FetchError; fetchError.status = response.status; fetchError.statusText = response.statusText; throw fetchError; }
        const data = await response.json(); const assistantResponse = data.answer || "Sorry, empty response."; addMessage(assistantResponse, 'assistant'); speak(assistantResponse); if (data.source) { console.log("Backend source:", data.source); }
    } catch (error: unknown) { /* ... error handling ... */
        setIsThinking(false); setStatusMessage(''); let errorMessage = 'An unknown error occurred.'; if (error instanceof Error) { errorMessage = error.name === 'AbortError' ? 'Request timed out.' : (error.message || 'Could not reach backend.'); } console.error('Send message error:', error); addMessage(`Error: ${errorMessage}`, 'system'); setStatusMessage(`Error: ${errorMessage}`);
    }
  }, [userInput, messages, addMessage, backendUrl, isThinking, speak, handleStopSpeaking]);

  // --- Speech Recognition Setup ---
  useEffect(() => { /* ... same setup logic ... */
    if (typeof window === 'undefined') return;
    let recognition: SpeechRecognition | null = null;
    const initializeRecognition = () => { const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition; if (!SpeechRecognitionAPI) { setStatusMessage('Speech recognition not supported.'); setIsSpeechRecognitionSupported(false); return null; } const r = new SpeechRecognitionAPI(); r.continuous = false; r.lang = 'en-US'; r.interimResults = false; r.maxAlternatives = 1; return r; };
    try {
      recognition = initializeRecognition(); if (!recognition) return; setIsSpeechRecognitionSupported(true); recognitionRef.current = recognition;
      recognition.onresult = (event: SpeechRecognitionEvent) => { const transcript = event.results[event.results.length - 1][0].transcript.trim(); if (transcript) { handleSendMessage(transcript); setIsListening(false); setStatusMessage(p => (p === 'Listening...' || p === 'Speech detected...') ? '' : p); } };
      recognition.onerror = (event: SpeechRecognitionErrorEvent) => { const err = event.error; console.error('SR error:', err, event.message); if (err === 'aborted') { console.log('SR aborted.'); } else { let msg = `Speech error: ${err}`; if (err === 'not-allowed' || err === 'service-not-allowed') msg = 'Mic permission denied.'; else if (err === 'no-speech') msg = 'No speech detected.'; else if (err === 'audio-capture') msg = 'Mic error.'; setStatusMessage(msg); } setIsListening(false); };
      recognition.onaudiostart = () => { setStatusMessage('Listening...'); setIsListening(true); };
      recognition.onend = () => { console.log('SR ended.'); setIsListening(false); setStatusMessage(p => (p === 'Listening...' || p === 'Speech detected...') ? '' : p); };
      recognition.onspeechstart = () => setStatusMessage('Speech detected...');
      recognition.onspeechend = () => console.log('Speech ended.');
    } catch (error) { console.error('SR init error:', error); setStatusMessage('SR init error'); setIsListening(false); setIsSpeechRecognitionSupported(false); }
    return () => { const r = recognitionRef.current; if (r) { r.onresult = null; r.onerror = null; r.onend = null; r.onaudiostart = null; r.onspeechstart = null; r.onspeechend = null; r.abort(); recognitionRef.current = null; } };
  }, [handleSendMessage]);

  // --- Auto-scroll messages ---
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  // --- More Core Functions ---
  const startListening = useCallback(() => { /* ... same logic ... */
      const r = recognitionRef.current; if (!isSpeechRecognitionSupported || !r || isListening || isThinking) { if (!isSpeechRecognitionSupported) setStatusMessage("SR not supported."); return; } handleStopSpeaking(); try { setUserInput(''); setStatusMessage('Initializing mic...'); r.start(); } catch (error) { console.error("SR start error:", error); if (error instanceof DOMException && error.name === 'InvalidStateError') { if (r) r.abort(); setIsListening(false); setStatusMessage('Mic ready. Try again.'); } else { setStatusMessage('Mic error.'); setIsListening(false); } }
  }, [isListening, isThinking, isSpeechRecognitionSupported, handleStopSpeaking]);

  const stopListening = useCallback(() => { /* ... same logic ... */
      const r = recognitionRef.current; if (!r || !isListening) return; try { r.abort(); } catch (error) { console.error("SR stop error:", error); setStatusMessage('Error stopping mic'); setIsListening(false); }
  }, [isListening]);

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => { /* ... same logic ... */
      if (event.key === 'Enter' && !event.shiftKey && !isThinking) { event.preventDefault(); handleSendMessage(); }
  };

  // --- JSX Rendering ---
  return (
    <div className="bg-gradient-to-br from-gray-100 via-white to-gray-200 dark:from-gray-800 dark:via-gray-900 dark:to-black flex items-center justify-center min-h-screen p-4 font-sans">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-2xl mx-auto overflow-hidden flex flex-col" style={{ height: 'calc(100vh - 4rem)', maxHeight: '800px' }}>
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 dark:from-blue-700 dark:to-indigo-800 text-white p-4 shadow-md flex-shrink-0">
          <h1 className="text-xl font-semibold text-center">Ask Srimanth&apos;s AI Assistant</h1>
        </div>

        {/* Chat Messages Area */}
        <div id="messages" className="flex-grow overflow-y-auto p-4 md:p-6 space-y-4 bg-gray-50 dark:bg-gray-900 scrollbar-thin scrollbar-thumb-blue-500 scrollbar-track-gray-200 dark:scrollbar-thumb-blue-700 dark:scrollbar-track-gray-700">
          {/* ... message mapping ... */}
          {messages.map((msg) => ( <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}> <div className={`p-3 rounded-lg max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg shadow-md break-words text-sm md:text-base ${ msg.sender === 'user' ? 'bg-blue-500 dark:bg-blue-600 text-white rounded-br-none' : msg.sender === 'assistant' ? 'bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-100 rounded-bl-none' : 'bg-red-100 dark:bg-red-900 border border-red-300 dark:border-red-700 text-red-700 dark:text-red-200 w-full text-center text-xs md:text-sm' }`}> {msg.text} </div> </div> ))}
          {/* ... thinking indicator ... */}
          {isThinking && ( <div className="flex justify-start"> <div className="p-3 rounded-lg shadow-md bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-100 rounded-bl-none animate-pulse text-sm md:text-base"> <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mr-1 animate-bounce" style={{animationDelay: '0ms'}} /> <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mr-1 animate-bounce" style={{animationDelay: '150ms'}} /> <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}} /> </div> </div> )}
          <div ref={messagesEndRef} className="h-1" />
        </div>

        {/* Status Bar */}
        <div className="px-4 py-1 text-xs md:text-sm text-center text-gray-500 dark:text-gray-400 h-6 flex-shrink-0">
          {statusMessage}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 dark:border-gray-700 p-3 md:p-4 bg-white dark:bg-gray-800 flex-shrink-0">
          <div className="flex items-center space-x-2">
            {/* ... input field ... */}
            <input type="text" id="user-input" aria-label="Chat input" placeholder={isListening ? "Listening..." : "Type your question or use the mic..."} value={userInput} onChange={(e) => setUserInput(e.target.value)} onKeyDown={handleKeyPress} disabled={isListening || isThinking} className="flex-grow p-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-600 transition duration-150 disabled:opacity-70 disabled:bg-gray-100 dark:disabled:bg-gray-700 dark:bg-gray-700 dark:text-gray-100 dark:placeholder-gray-400" />
            {/* ... send button ... */}
            <button onClick={() => handleSendMessage()} title="Send Message" aria-label="Send Message" disabled={!userInput.trim() || isListening || isThinking} className="bg-blue-500 hover:bg-blue-600 dark:bg-blue-600 dark:hover:bg-blue-700 text-white p-3 rounded-lg transition duration-150 shadow focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-600 focus:ring-offset-1 dark:focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"> <Send size={20} /> </button>
            {/* ... mic button ... */}
            {isSpeechRecognitionSupported && ( <button onClick={isListening ? stopListening : startListening} title={isListening ? "Stop Listening" : "Ask with Voice"} aria-label={isListening ? "Stop Listening" : "Ask with Voice"} disabled={isThinking} className={`p-3 rounded-lg transition duration-150 shadow focus:outline-none focus:ring-2 focus:ring-offset-1 dark:focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed ${ isListening ? 'bg-red-500 hover:bg-red-600 dark:bg-red-600 dark:hover:bg-red-700 text-white focus:ring-red-500 dark:focus:ring-red-600 animate-pulse' : 'bg-indigo-500 hover:bg-indigo-600 dark:bg-indigo-600 dark:hover:bg-indigo-700 text-white focus:ring-indigo-500 dark:focus:ring-indigo-600' }`}> {isListening ? <Square size={20} /> : <Mic size={20} />} </button> )}
            {/* ... stop speaking button ... */}
            {isSpeaking && ( <button onClick={handleStopSpeaking} title="Stop Speaking" aria-label="Stop Speaking" className="p-3 rounded-lg transition duration-150 shadow focus:outline-none focus:ring-2 focus:ring-offset-1 dark:focus:ring-offset-gray-800 bg-yellow-500 hover:bg-yellow-600 dark:bg-yellow-600 dark:hover:bg-yellow-700 text-white focus:ring-yellow-500 dark:focus:ring-yellow-600"> <StopCircle size={20} /> </button> )}
          </div>
        </div>
      </div>
    </div>
  );
}