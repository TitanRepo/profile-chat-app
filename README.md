# Profile Chat Application

An interactive chat interface that allows users to have conversations about professional experience and skills using voice or text input. The application uses Next.js for the frontend and Flask for the backend, powered by Google's Gemini AI for natural language processing.

## Features

- 🎯 Interactive chat interface with AI responses
- 🎤 Voice input support with speech-to-text conversion
- 🔊 Text-to-speech for AI responses
- 💬 Real-time chat experience
- 🎨 Modern, responsive UI with gradients and animations
- 🔒 Environment variable support for secure API key management

## Tech Stack

### Frontend
- Next.js 14+ with TypeScript
- TailwindCSS for styling
- Web Speech API for voice interactions
- Lucide React for icons

### Backend
- Flask (Python)
- Google Gemini AI for natural language processing
- CORS support for cross-origin requests
- Environment variable management with python-dotenv

## Prerequisites

Before running the application, make sure you have:
- Node.js 18+ installed
- Python 3.8+ installed
- Google Gemini API key
- Resume data in JSON format

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/TitanRepo/profile-chat-app.git
cd profile-chat-app
```

2. Frontend Setup:
```bash
cd my-profile-app
npm install
```

3. Backend Setup:
```bash
cd backend
pip install -r requirements.txt
```

4. Environment Configuration:
- Create a `.env` file in the backend directory with:
```
GEMINI_API_KEY=your_api_key_here
```

5. Resume Data:
- Ensure `resume_data.json` is present in the backend directory

## Running the Application

1. Start the backend server:
```bash
cd backend
python app.py
```
The backend will run on `http://localhost:5000`

2. Start the frontend development server:
```bash
cd my-profile-app
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the application

## Features in Detail

- **Chat Interface**: Type or speak your questions about the profile
- **Voice Input**: Click the microphone icon to start voice input
- **Text-to-Speech**: AI responses are read aloud automatically
- **Responsive Design**: Works seamlessly on both desktop and mobile devices

## Development

The project uses ESLint for code quality and TypeScript for type safety. To run the development tools:

```bash
# Run ESLint
npm run lint

# Type checking
npm run type-check
```

## Deployment

The frontend can be easily deployed on Vercel, and the backend can be deployed on any Python-supporting platform like Heroku, Google Cloud Platform, or AWS.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
