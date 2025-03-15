#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:04:28 2025

@author: alexyu
"""

import discord
from discord.ext import commands
import logging
import signal
import re
import asyncio
from collections import Counter
import pandas as pd
import random
import nest_asyncio

# Setup logging and nest_asyncio
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    logging.info(f'Logged in as {bot.user}')
    
@bot.event
async def on_disconnect():
    await bot.http.session.close()

# Load data
material = []
with open("desktop/illegal_bq/luke/luke_plain.txt", "r") as file:
    info = file.read()

verse = info.splitlines()
for x in verse:

    verse_ref = x.split(":",3)
    verse_ref = [int(x) if x.isdigit() else x for x in verse_ref]
    
    quote = verse_ref[3].replace('\xa0', '').strip()
    newverse = dict({"book": verse_ref[0], "chapter": verse_ref[1], "verse": verse_ref[2], "quote": quote})
    if material.count(newverse) < 1:
        material.append(newverse)
    else:
        break

df = pd.DataFrame(material)
pd.set_option('display.max_colwidth', None)

# Find verse based on reference
def versequery(book, chapter, verse):
    quote = df[(df['book'] == book) & (df['chapter'] == chapter) & (df['verse'] == verse)]
    quote = quote[['quote']]
    quote = quote.to_string(index=False, header=False)
    return quote + '\n(' + book + ' ' + str(chapter) + ':' + str(verse) + ')'

# Find verse based on keywords
def wordquery(words, limit = 10):
    if not words:
        return f"https://tenor.com/view/think-meme-thinking-memes-memes-2024-gif-6703217797690493255"
    
    # If `words is a single string, split it into a list of words
    if isinstance(words, str):
        words = words.split()
    
    # Create a regex pattern to match the entire phrase (all words together)
    phrase_pattern = r'\b' + re.escape(" ".join(words)) + r'\b'
    
    # Find rows where the quote contains the entire phrase (case-insensitive)
    matches = df[df['quote'].str.contains(phrase_pattern, case=False, regex=True)]
    
    # If no matches are found, return a message
    if matches.empty:
        return f"No quotes found containing the phrase: '{' '.join(words)}'"
    
    # Extract book, chapter, and verse information for the matching rows
    to_query = matches[['book', 'chapter', 'verse']].values.tolist()
    
    # Query and collect results for each match (up to the specified limit)
    results = []
    for i, values in enumerate(to_query):
        if i >= limit:  # Stop after reaching the limit
            return f"Way too many"
        book, chapter, verse = values
        result = versequery(book, chapter, verse)
        
        # Bold the entire phrase in the result (case-insensitive)
        result = re.sub(phrase_pattern, r"**\g<0>**", result, flags=re.IGNORECASE)
        results.append(result)
    
    # Return all results as a single string
    return "\n\n".join(results)
    
# Create MA question
def multans():
    # MA docs link
    url = f"https://docs.google.com/spreadsheets/d/e/2PACX-1vSeZ1WU8cIu_viDmGGMypjDOaollcERBgvvVfGyeblh_bLLZ5Lagi8TMn-4t1De8sch2LGY2S99K119/pub?gid=1359126200&single=true&output=csv"
    
    # Create a list of questions
    ma_list = pd.read_csv(url)
    pool = ma_list["Question"].tolist()
    
    # Select a random index
    seed = random.randint(1,len(ma_list) - 1)
    select_prompt = pool[seed].split('>>')
    ans = ma_list.iat[seed,2]
    
    # TEMPORARY CODE
    ref = ma_list.iat[seed,0].replace(' ',':').split(':')
    ref = [int(x) if x.isdigit() else x for x in ref]
    book, chapter, verse = ref
    book = 'Luke'

    # Generate prompt
    prompt = select_prompt[0] + f"||{select_prompt[1]}||"
    prompt += f"\n\n**Answer:**\n||{ans}||\n\n**Verse:**\n||{versequery(book,chapter,verse)}||"
    
    return prompt
    
# Find a random one word key
def keyword1():
    # Split the text into words
    combined_text = df['quote'].str.cat(sep=' ')
    combined_text = re.sub(r'—', ' ', combined_text)   
    preparse = combined_text.split()
    
    # Clean punctuation 
    pattern = r'[—,!?;:.“‘’”\()]' 
    cleaned_words = [re.sub(pattern, '', word) for word in preparse]
    
    # Normalize case for counting
    lower_words = [word.lower() for word in cleaned_words]
    
    # Count word occurrences
    word_counts = Counter(lower_words)
    
    # Filter words that appear only once
    unique_lower_words = {word for word, count in word_counts.items() if count == 1}
    
    # Return the original words that are unique
    pool = [word for word in cleaned_words if word.lower() in unique_lower_words]
    pool.pop(-40)
    
    return pool
    
# Find a two word key
def keyword2():
    # Combine data to string
    combined_text = df['quote'].str.cat(sep=' ')
    combined_text = re.sub(r'—', ' ', combined_text)

    # Iterate through each keyword and replace it with an placeholder
    pattern = r'\b(?:' + '|'.join(map(re.escape, keyword1())) + r')\b'
    result = re.sub(pattern, '&&', combined_text, flags=re.IGNORECASE).split()
    
    # combine all words into phrases of two words and replace phrases with breaks in flow with a placeholder
    phrase = [' '.join(result[x:x+2]) for x in range(len(result)-1)]
    clean_phrase = [re.sub(r'\b(\w+)([^\w\s]+\s+)(\w+)\b', '&&', word) for word in phrase]
    
    # Remove placeholders from the list and clean punctuation
    remove_keys = pd.Series(clean_phrase)
    matches = remove_keys[~remove_keys.str.contains('&&')]
    no_keys = matches.values.tolist()
    pattern = r'[—,!?;:.“‘’”\()]' 
    clean_punc = [re.sub(pattern, '', word) for word in no_keys]

    # Find all phrases that only occur once
    lower_words = [word.lower() for word in clean_punc]    
    word_counts = Counter(lower_words)
    unique_lower_words = {word for word, count in word_counts.items() if count == 1}
    pool = [word for word in clean_punc if word.lower() in unique_lower_words]

    return pool

# Find a three word key
def keyword3():
    # Combine data to string
    combined_text = df['quote'].str.cat(sep=' ')
    combined_text = re.sub(r'—', ' ', combined_text)
    
    # Iterate through each keyword and replace it with an placeholder
    pattern1 = r'\b(?:' + '|'.join(map(re.escape, keyword1())) + r')\b'
    pattern2 = r'\b(?:' + '|'.join(map(re.escape, keyword2())) + r')\b'
    result = re.sub(pattern1, '&&', combined_text, flags=re.IGNORECASE)
    result = re.sub(pattern2, '&&', result, flags=re.IGNORECASE).split()
    
    # combine all words into phrases of three words and replace phrases with breaks in flow with a placeholder
    phrase = [' '.join(result[x:x+3]) for x in range(len(result)-2)]
    clean_phrase = [re.sub(r'[^\w\s\'\-]', '&&', word) for word in phrase]

    # Remove placeholders from the list and clean punctuation
    remove_keys = pd.Series(clean_phrase)
    matches = remove_keys[~remove_keys.str.contains('&&')]
    no_keys = matches.values.tolist()
    pattern = r'[—,!?;:.“‘’”\()]' 
    clean_punc = [re.sub(pattern, '', word) for word in no_keys]
    
    
    phrase_to_original = {phrase.lower(): phrase for phrase in clean_punc}
    phrase_counts = Counter(phrase.lower() for phrase in clean_punc)
    pool = [phrase_to_original[phrase] for phrase, count in phrase_counts.items() if count == 1]
 
    return pool

'''
def cvr():
    
def quote():
    
def sit():
'''

@bot.command()
async def kw(ctx, n=0):
    if n == 0:
        n = random.randint(1,3)
    if n == 1:
        prompt = random.choice(keyword1())
    elif n == 2:
        prompt = random.choice(keyword2())
    elif n == 3:
        prompt = random.choice(keyword3())
    else: 
        await ctx.send(f"https://tenor.com/view/nuh-uh-beocord-no-lol-gif-24435520")
        return
    
    prompt += f"\n\n||{wordquery(prompt)}||\n"
    await ctx.send(prompt)

@bot.command()
async def f(ctx, *, word=""):
    await ctx.send(wordquery(word))
    
@bot.command()
async def ma(ctx):
    await ctx.send(multans())

# Run the bot
async def main():
    await bot.start(BOT_TOKEN)
    
# Handle shutdown signals
def shutdown_signal_handler(signal, frame):
    logging.info("Data saved. Bot is shutting down.")
    loop = asyncio.get_event_loop()
    loop.create_task(bot.close())
    loop.stop()
    
signal.signal(signal.SIGINT, shutdown_signal_handler)
signal.signal(signal.SIGTERM, shutdown_signal_handler)

# Start the bot
loop = asyncio.get_event_loop()
loop.run_until_complete(main())

