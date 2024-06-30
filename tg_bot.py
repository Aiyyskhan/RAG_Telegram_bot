import os
from dotenv import load_dotenv
import telebot
import rag as rag

load_dotenv() #"env/.env")

telebot_api_key = os.getenv("TELEBOT_API_KEY")

bot = telebot.TeleBot(telebot_api_key)

previous_questions = []

@bot.message_handler(func=lambda _: True)
def handle_message(msg):
	
	res = rag.main(msg.text, msg.from_user.id, "T-800")

	bot.send_message(chat_id=msg.from_user.id, text=res)

bot.infinity_polling()
# bot.polling()