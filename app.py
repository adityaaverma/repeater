import os
import assemblyai as aai
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Initialize the Google Generative AI model
os.environ['GOOGLE_API_KEY']="AIzaSyCeMQ1-ACDbQ6kmqaxd4QRo12G1Fu0T9QM"
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

config = {"configurable": {"session_id": "abc5"}}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are a helper who helps students find properties according to their requirements as they go abroad for studies. Be resourceful and efficient.''',
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model
with_message_history = RunnableWithMessageHistory(chain, get_session_history)


class AI_Assistant:
    def __init__(self):
        # Use os.getenv to get API keys from the environment
        self.client = ElevenLabs(
            api_key='sk_31d6ac881d1ed1ae54d241d40a91b301b78b7b9ca46df110', # Defaults to ELEVEN_API_KEY
        )
        aai.settings.api_key = '1949c93b35c147b7b79f5cfe5113e3f8'
        self.transcriber = None
        # self.full_transcript = [
        #     {"role": "system", "content": "You are a receptionist at a dental clinic. Be resourceful and efficient."},
        # ]
    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate = 16000,
            on_data = self.on_data,
            on_error = self.on_error,
            on_open = self.on_open,
            on_close = self.on_close,
            end_utterance_silence_threshold = 1000
        )

        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate =16000)
        self.transcriber.stream(microphone_stream)

    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        print("An error occurred:", error)

    def on_close(self):
        print("Transcription session closed.")

    def generate_ai_response(self, transcript):
        self.stop_transcription()

        print(f"\nEnquirer: {transcript.text}", end="\r\n")

        response = with_message_history.invoke(
            [HumanMessage(content=transcript.text)],
            config=config,
        )

        ai_response = response.content
        self.generate_audio(ai_response)

        self.start_transcription()
        print(f"\nReal-time transcription resumed.")

    def generate_audio(self, text):
        # self.full_transcript.append({"role": "assistant", "content": text})
        print(f"\nAI Receptionist: {text}")

        audio = self.client.generate(
        text=text,
        voice="Rachel",
        model="eleven_multilingual_v2"
        )

        # Stream the generated audio
        play(audio)


# Initialize and start AI Assistant
greeting = "Thank you for calling University Living. How can I help you today?"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()
