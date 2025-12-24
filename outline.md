
class Transcribe

    callbacks:
        async def on_realtime_update(self, text: str):
        async def on_realtime_stabilized(self, text: str):
        async def on_final_transcription(self, user_message: str):

    functions:
        def transcription_loop(self):
            user_message = self.recorder.text()

        def feed_audio(self, audio_bytes: bytes):
            self.recorder.feed_audio(audio_bytes, original_sample_rate=16000)


class WebSocketManager

    functions:
        async def handle_text_message(self, message: str):

        async def handle_audio_message(self, audio_bytes: bytes):
            self.transcribe.feed_audio(audio_bytes)


class Chat

    functions:
        async def send_conversation_prompt(self, messages: str, character: Character, user_name: str = "Jay", model_settings: ModelSettings = None):
        async def character_response_stream(self, character: Character, text_stream: AsyncIterator) -> str:
        async def chunk_generator() -> AsyncGenerator[str, None]:
        async for sentence in generate_sentences_async(chunk_generator()):
            await sentence_queue.put(sentence)


Class ChatConversation


