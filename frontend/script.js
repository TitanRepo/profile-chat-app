// script.js
const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const micButton = document.getElementById('mic-button');
const statusDiv = document.getElementById('status');

const backendUrl = 'http://127.0.0.1:5000/query'; // Your Flask backend URL

// --- Web Speech API Setup ---
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition;
let isListening = false;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false; // Stop after first detection of speech ending
    recognition.lang = 'en-US'; // Set language
    recognition.interimResults = false; // Get final result only
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        userInput.value = transcript; // Put transcribed text into input box
        sendMessage(transcript); // Send the message automatically
        stopListening();
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        statusDiv.textContent = `Error: ${event.error}`;
        stopListening();
    };

    recognition.onend = () => {
        if (isListening) { // Handle unexpected stops
            stopListening();
        }
    };

} else {
    micButton.disabled = true;
    statusDiv.textContent = 'Speech recognition not supported in this browser.';
}

// --- Event Listeners ---
sendButton.addEventListener('click', () => {
    const messageText = userInput.value.trim();
    if (messageText) {
        sendMessage(messageText);
    }
});

userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        const messageText = userInput.value.trim();
        if (messageText) {
            sendMessage(messageText);
        }
    }
});

micButton.addEventListener('click', () => {
    if (!recognition) return;

    if (isListening) {
        stopListening();
    } else {
        startListening();
    }
});

// --- Functions ---
function addMessage(text, sender) {
    const messageElem = document.createElement('div');
    messageElem.classList.add('message', sender);
    messageElem.textContent = text;
    messagesDiv.appendChild(messageElem);
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to bottom
}

async function sendMessage(messageText) {
    addMessage(messageText, 'user');
    userInput.value = ''; // Clear input field

    try {
        const response = await fetch(backendUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: messageText }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const assistantResponse = data.answer || "Sorry, I couldn't get a response.";
        addMessage(assistantResponse, 'assistant');
        speak(assistantResponse); // Speak the response

    } catch (error) {
        console.error('Error sending message:', error);
        addMessage(`Error: Could not reach the backend. ${error.message}`, 'assistant');
    }
}

function startListening() {
    if (!recognition || isListening) return;
    try {
        recognition.start();
        isListening = true;
        micButton.textContent = '🛑'; // Change icon/text to indicate listening
        micButton.classList.add('listening');
        statusDiv.textContent = 'Listening...';
    } catch (error) {
         console.error("Error starting recognition:", error);
         statusDiv.textContent = 'Mic error. Please ensure permission is granted.';
         isListening = false; // Make sure state is reset
    }
}

function stopListening() {
    if (!recognition || !isListening) return;
    recognition.stop();
    isListening = false;
    micButton.textContent = '🎤';
    micButton.classList.remove('listening');
    statusDiv.textContent = '';
}

// Text-to-Speech using Web Speech API
function speak(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        // Optional: Configure voice, rate, pitch
        // const voices = window.speechSynthesis.getVoices();
        // utterance.voice = voices[/* choose a voice */];
        utterance.rate = 1; // Speed (0.1 to 10)
        utterance.pitch = 1; // Pitch (0 to 2)
        window.speechSynthesis.speak(utterance);
    } else {
        console.warn('Speech synthesis not supported in this browser.');
    }
}

// Optional: Preload voices for TTS if needed
if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => {
        // Voices loaded - you could update a voice selection UI here
        console.log("Speech synthesis voices loaded.");
    };
}