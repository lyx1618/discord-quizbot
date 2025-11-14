#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:04:28 2025

@author: alexyu
"""

import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio
import logging
import signal
import re
from collections import defaultdict
import pandas as pd
import random
import nest_asyncio


# Setup logging and nest_asyncio
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)

# Bot setup
intents = discord.Intents.default()
client = discord.Client(intents=intents)
intents.message_content = True
intents.members = True
intents.messages = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

@client.event
async def on_ready():
    logging.info(f'Logged in as {bot.user}')
    
@client.event
async def on_disconnect():
    await bot.http.session.close()
    
# Find verse based on reference
def versequery(book, chapter, verse):
    quote = df[(df['book'] == book) & (df['chapter'] == chapter) & (df['verse'] == verse)]
    quote = quote[['quote']]
    quote = quote.to_string(index=False, header=False)
    return quote + '\n(' + book + ' ' + str(chapter) + ':' + str(verse) + ')'

# Find verse based on words
def wordquery(words, limit=10, chapter=None):
    if not words:
        return f"https://tenor.com/view/think-meme-thinking-memes-memes-2024-gif-6703217797690493255"
    
    # If words is a single string, split it into a list of words
    if isinstance(words, str):
        words = words.split()
    
    # Build regex pattern for each word, handling hyphens
    word_patterns = []
    for word in words:
        if word.endswith('-'):
            # For partial words (ending with hyphen), match as prefix
            base_word = re.escape(word[:-1])
            word_patterns.append(rf'\b{base_word}\w*')
        else:
            # For complete words, match exactly
            word_patterns.append(rf'\b{re.escape(word)}\b')
    
    # Combine patterns with word boundaries and optional whitespace
    phrase_pattern = r'\s+'.join(word_patterns)
    
    # Start with all rows that match the phrase
    matches = df[df['quote'].str.contains(phrase_pattern, case=False, regex=True)]
    
    # If a specific chapter is requested, filter further
    if chapter is not None:
        matches = matches[matches['chapter'] == chapter]
        
    if matches.empty:
        if chapter is not None:
            return f"No quotes found matching '{' '.join(words)}' in chapter {chapter}"
        return f"No quotes found matching: '{' '.join(words)}'"
    
    # Extract book, chapter, and verse information
    to_query = matches[['book', 'chapter', 'verse']].values.tolist()
    
    # Query and collect results
    results = []
    for i, values in enumerate(to_query):
        if i >= limit:
            return f"Way too many"
        book, chapter, verse = values
        result = versequery(book, chapter, verse)
        
        # Find the exact match in the result using the same regex
        match = re.search(phrase_pattern, result, flags=re.IGNORECASE)
        if match:
            # Bold only the complete matched phrase
            matched_phrase = match.group(0)
            result = result.replace(matched_phrase, f"**{matched_phrase}**")
        results.append(result)
    
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
    '''
    # TEMPORARY CODE
    ref = ma_list.iat[seed,0].replace(' ',':').split(':')
    ref = [int(x) if x.isdigit() else x for x in ref]
    book, chapter, verse = ref
    book = 'Luke'
    '''
    # Generate prompt
    prompt = select_prompt[0] + f"||{select_prompt[1]}||"
    prompt += f"\n\n**Answer:**\n||{ans}||\n\n**Verse:**\n||{versequery(book,chapter,verse)}||"
    
    return prompt

# Create CR Question ***UNFINISHED***
def chapref(chapter=None):
    # CR docs link
    url = f"https://docs.google.com/spreadsheets/d/e/2PACX-1vSyf5UzUoh-9sSkXs49qbRntKc9sqQbJ3JmiM3usseHN1DCkK-FcSFCBUMwmgsjQi0xQ5oRDS04OJEB/pub?gid=1488963601&single=true&output=csv"
    
    cr_list = pd.read_csv(url)
    pool = cr_list["Question"].tolist()
    
    # Filter by chapter if specified
    if chapter is not None:
       # Extract book and chapter
       ref_parts = cr_list["Reference"].str.split(r'[: ]', expand=True)
       cr_list = cr_list[ref_parts[1] == str(chapter)]  # Filter to specific chapter

    # Random selection
    seed = random.randint(0, len(cr_list) - 1)
    selected_row = cr_list.iloc[seed]
    
    # Split question and get answer
    select_prompt = selected_row["Question"].split('>>')
    ans = selected_row.iloc[3]   
    '''
    # TEMPORARY CODE
    ref = selected_row.iloc[0].replace(' ',':').split(':')
    ref = [int(x) if x.isdigit() else x for x in ref]
    book, chapter, verse = ref
    book = 'Luke'
    '''
    # Generate prompt
    prompt = select_prompt[0] + f"||{select_prompt[1]}||"
    prompt += f"\n\n**Answer:**\n||{ans}||\n\n**Verse:**\n||{versequery(book,chapter,verse)}||"

    return prompt

def findkeys():
    
    word_re = re.compile(r"[0-9A-Za-z]+(?:['’][0-9A-Za-z]+)*", flags=re.UNICODE)
    token_re = re.compile(r"[0-9A-Za-z]+(?:['’][0-9A-Za-z]+)*|[^\w\s'’]", flags=re.UNICODE)

    # Collect occurrences globally: mapping norm_phrase -> list of (book,chapter,verse,start,end)
    occs_1 = defaultdict(list)
    occs_2 = defaultdict(list)
    occs_3 = defaultdict(list)

    # Keep the original verse text for slicing later (keyed by (book,chapter,verse))
    verse_texts = {}

    # First pass: iterate verses and gather all ngram occurrences (with per-verse offsets)
    for _, row in df.iterrows():
        book = row['book']
        chapter = row['chapter']
        verse = row['verse']
        quote = row['quote']
        verse_key = (book, chapter, verse)
        verse_texts[verse_key] = quote

        # tokenize with positions in this verse
        tokens = []
        for m in token_re.finditer(quote):
            tok = m.group(0)
            if word_re.fullmatch(tok):
                norm = tok.replace("’", "'").lower()
                tokens.append({
                    'norm': norm,
                    'text': tok,
                    'start': m.start(),
                    'end': m.end()
                })
            else:
                tokens.append({'punct': tok, 'start': m.start(), 'end': m.end()})

        # split into segments by punctuation
        segments = []
        curr = []
        for t in tokens:
            if 'norm' in t:
                curr.append(t)
            else:
                if curr:
                    segments.append(curr)
                    curr = []
        if curr:
            segments.append(curr)

        # build ngrams and register occurrences
        for seg in segments:
            L = len(seg)
            for i in range(L):
                n1 = seg[i]['norm']
                occs_1[n1].append((book, chapter, verse, seg[i]['start'], seg[i]['end']))
            for i in range(L-1):
                norm2 = seg[i]['norm'] + ' ' + seg[i+1]['norm']
                occs_2[norm2].append((book, chapter, verse, seg[i]['start'], seg[i+1]['end']))
            for i in range(L-2):
                norm3 = seg[i]['norm'] + ' ' + seg[i+1]['norm'] + ' ' + seg[i+2]['norm']
                occs_3[norm3].append((book, chapter, verse, seg[i]['start'], seg[i+2]['end']))

    # Compute global counts
    one_counts = {k: len(v) for k, v in occs_1.items()}
    two_counts = {k: len(v) for k, v in occs_2.items()}
    three_counts = {k: len(v) for k, v in occs_3.items()}

    # Keep only globally-unique occurrences (count==1)
    uniq_one = {k: v[0] for k, v in occs_1.items() if one_counts.get(k, 0) == 1}   # map norm -> single occ tuple
    uniq_two = {k: v[0] for k, v in occs_2.items() if two_counts.get(k, 0) == 1}
    uniq_three = {k: v[0] for k, v in occs_3.items() if three_counts.get(k, 0) == 1}

    # Exclusivity rules (global)
    one_norms = set(uniq_one.keys())
    # filter two such that none of its words are a unique one-word
    two_filtered = {k: v for k, v in uniq_two.items() if not any(w in one_norms for w in k.split())}
    two_norms = set(two_filtered.keys())

    three_filtered = {}
    for k, v in uniq_three.items():
        w0, w1, w2 = k.split()
        pair1 = f"{w0} {w1}"
        pair2 = f"{w1} {w2}"
        if (w0 not in one_norms and w1 not in one_norms and w2 not in one_norms
            and pair1 not in two_norms and pair2 not in two_norms):
            three_filtered[k] = v

    # Build rows for DataFrame: slice the original verse text using saved offsets
    rows = []
    for norm, occ in uniq_one.items():
        book, chapter, verse, s, e = occ
        verse_key = (book, chapter, verse)
        phrase_text = verse_texts[verse_key][s:e]
        rows.append({
            'book': book, 'chapter': chapter, 'verse': verse,
            'phrase': phrase_text, 'norm': norm, 'start': s, 'end': e, 'length': 1
        })

    for norm, occ in two_filtered.items():
        book, chapter, verse, s, e = occ
        verse_key = (book, chapter, verse)
        phrase_text = verse_texts[verse_key][s:e]
        rows.append({
            'book': book, 'chapter': chapter, 'verse': verse,
            'phrase': phrase_text, 'norm': norm, 'start': s, 'end': e, 'length': 2
        })

    for norm, occ in three_filtered.items():
        book, chapter, verse, s, e = occ
        verse_key = (book, chapter, verse)
        phrase_text = verse_texts[verse_key][s:e]
        rows.append({
            'book': book, 'chapter': chapter, 'verse': verse,
            'phrase': phrase_text, 'norm': norm, 'start': s, 'end': e, 'length': 3
        })

    df_phrases = pd.DataFrame(rows)

    # Sort by book/chapter/verse then start
    df_phrases = df_phrases.sort_values(['book','chapter','verse','start']).reset_index(drop=True)
    return df_phrases

def parse_reference(ref_text):
    if not ref_text or not ref_text.strip():
        # Whole material
        return [
            {"book": "1 Corinthians", "chapters": None},
            {"book": "2 Corinthians", "chapters": None},
        ]
    
    ref_text = ref_text.strip().lower()
    parts = re.split(r"\s*,\s*", ref_text)  # split by commas for multi-book parts
    
    refs = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Determine book
        if part.startswith("1c"):
            book = "1 Corinthians"
            body = part[2:].strip()
        elif part.startswith("2c"):
            book = "2 Corinthians"
            body = part[2:].strip()
        else:
            raise ValueError(f"Invalid book specifier in '{part}' (expected 1c or 2c)")
        
        if not body:
            refs.append({"book": book, "chapters": None})
            continue
        
        # Split on '/' for separate chapter or ranges
        segments = re.split(r"/", body)
        chapters = set()
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            m = re.match(r"^(\d+)-(\d+)$", seg)
            if m:
                start, end = map(int, m.groups())
                chapters.update(range(start, end + 1))
            elif seg.isdigit():
                chapters.add(int(seg))
            else:
                raise ValueError(f"Bad chapter segment '{seg}'")
        
        refs.append({"book": book, "chapters": sorted(chapters) if chapters else None})
    
    return refs

def select_phrases(df_phrases, ref_text=None):
    parsed_refs = parse_reference(ref_text)

    frames = []
    for r in parsed_refs:
        book = r["book"]
        chaps = r["chapters"]
        df_book = df_phrases[df_phrases["book"] == book]
        if chaps:
            df_book = df_book[df_book["chapter"].isin(chaps)]
        frames.append(df_book)
    
    if not frames:
        return df_phrases.copy()
    
    return pd.concat(frames, ignore_index=True)

def finish():
    pool = df[df['rd300'] == 'T']['quote'].tolist()
    preparse = []
    for verse in pool:
        split_five = verse.split(" ",5)
        first_five = split_five[:-1]
       
        pattern = r'[—,!?;:.“‘’”\()]' 
        cleaned_words = [re.sub(pattern, '', word) for word in first_five]
        preparse.append(' '.join(cleaned_words))
        
    results, no_split = [], []
    for phrase in preparse:
        words = phrase.split()
        n = len(words)
        
        found_unique = False
        for k in range(1, 6):
            sub = " ".join(words[:k])
            count = sum(p.startswith(sub) for p in preparse)
            
            if count == 1:  # unique prefix found
            
                # build phrase with separator
                rest = " ".join(words[k:])
                if rest:
                    results.append(sub + " // " + rest)
                else:
                    results.append(sub)  # nothing after it
                found_unique = True
                break
        
        if not found_unique:
            # discard this phrase
            pass
    
    return results
    
'''
def cvr():
    
def quote():
    
def sit():
'''

@bot.command()
async def kw(ctx, *, ref: str=None):
    subset = select_phrases(findkeys(), ref)
    if subset.empty:
        await ctx.send("https://tenor.com/view/nuh-uh-beocord-no-lol-gif-24435520")
        return
    
    prompt = random.choice(subset['phrase'].tolist())
    prompt += f"\n\n||{wordquery(prompt)}||\n"
    
    #TO TEST CODE
    '''
    prompt = "\n".join(
        f"{r.book} {r.chapter}:{r.verse} → {r.phrase}"
        for r in subset.head(40).itertuples()
    )
    ''' 
    await ctx.send(prompt)

@bot.command()
async def ftv(ctx):
    p = random.choice(finish())
    
    prompt = p.replace('//','>> ||') + '||'
    prompt += f"\n\n||{wordquery(p.replace('//',''))}||\n"
    await ctx.send(prompt)
    
@bot.command()
async def f(ctx, *, args=""):
    # Split input into search terms and optional chapter number
    parts = args.split()
    chapter = None
    
    # Check if last part is a number (chapter)
    if len(parts) > 1 and parts[-1].isdigit():
        chapter = int(parts[-1])
        words = ' '.join(parts[:-1])
    else:
        words = args
    
    # Handle empty input
    if not words.strip():
        await ctx.send(f"https://tenor.com/view/ken-jeong-community-too-small-to-read-read-reading-gif-5494204")
        return
    
    # Process the query
    result = wordquery(words, chapter=chapter)
    await ctx.send(result)
    
@bot.command()
async def ma(ctx):
    await ctx.send(multans())

@bot.command()
async def cr(ctx, args=""):

    try: 
        chapter = int(args)
    except: 
        chapter = None
    
    # Process the query
    result = chapref(chapter=chapter)
    await ctx.send(result)    

@bot.command()
async def help(ctx):
   help_message = '''
   
    **Bot Commands:**
    
    1. **!help** - Displays this help message
    2. **!f {phrase}** - Shows the verse/reference of all instances of that phrase occuring in the text.
    3. **!kw {book} {chapter}** - Returns a random one, two, or three word key and its verse/reference. Can be filtered by chapter (eg. 1-4, 3/5)
    4. **!ma** - Returns a random multiple answer question from the doc.
    5. **!ftv** - Returns a random club 300 finish.
    '''
    
   await ctx.send(help_message)

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.lower() == "ping":
        await message.channel.send("pong!")

material = []
with open("home/yuxela/quizbot/corinthians.txt", "r", encoding="utf-8") as file:
    info = file.read()

verse = info.splitlines()
for x in verse:
    verse_ref = x.split(";", 4)
    verse_ref = [int(v) if v.isdigit() else v for v in verse_ref]
    if len(verse_ref) < 5:
        continue
    quote = verse_ref[4].replace('\xa0', '').strip()
    book = '1 Corinthians' if verse_ref[0].replace('\ufeff', '').strip() == '1c' else '2 Corinthians'
    newverse = dict({
        "book": book,
        "chapter": verse_ref[1],
        "verse": verse_ref[2],
        "rd300": verse_ref[3],
        "quote": quote
    })
    if material.count(newverse) < 1:
        material.append(newverse)
    else:
        break

df = pd.DataFrame(material)
pd.set_option('display.max_colwidth', None)
text = df["quote"].str.cat(sep=' ')


# Run the bot
async def main():
    await bot.start(TOKEN)
    
# Handle shutdown signals  
def shutdown_signal_handler(signal, frame):
    logging.info("Force shutting down!")
    # Raise SystemExit to terminate immediately
    raise SystemExit(0)
    
signal.signal(signal.SIGINT, shutdown_signal_handler)
signal.signal(signal.SIGTERM, shutdown_signal_handler)

# Start the bot
asyncio.run(main())
