import emoji
import re
import contractions
import langdetect
import string
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def replaceEmoji(text):
    text = emoji.replace_emoji(text,replace = "");
    return text

def stripUnwantedEntities(text):
    text = re.sub(r'\r|\n', ' ', text.lower()) 
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def cleanHashtags(tweet):
    new_tweet = re.sub(r'(\s+#[\w-]+)+\s*$', '', tweet).strip()
    new_tweet = re.sub(r'#([\w-]+)', r'\1', new_tweet).strip()
    return new_tweet

def filterChars(text):
    return ' '.join('' if ('$' in word) or ('&' in word) else word for word in text.split())

def removeMultipleSpaces(text):
    return re.sub(r"\s\s+", " ", text)

def filterNonEnglish(text):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    return text if lang == "en" else ""

def expandContractions(text):
    return contractions.fix(text)

def removeNumbers(text):
    return re.sub(r'\d+', '', text)

def lemmatize(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def removeShortWords(text, min_len=2):
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)

def replaceElongatedWords(text):
    regex_pattern = r'\b(\w+)((\w)\3{2,})(\w*)\b'
    return re.sub(regex_pattern, r'\1\3\4', text)

def removeRepeatedPunctuation(text):
    return re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)

def removeExtraWhitespace(text):
    return ' '.join(text.split())

def removeURLShorteners(text):
    return re.sub(r'(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S+', '', text)

def removeSpacesTweets(tweet):
    return tweet.strip()
    
def removeShortTweets(tweet, min_words=3):
    words = tweet.split()
    return tweet if len(words) >= min_words else ""
def finalfunction(text):
    text = replaceEmoji(text);
    text = stripUnwantedEntities(text);
    text = cleanHashtags(text);
    text = filterChars(text);
    text = removeMultipleSpaces(text);
    text = filterNonEnglish(text);
    text = expandContractions(text);
    text = removeNumbers(text);
    text = lemmatize(text);
    text = removeShortWords(text);
    text = replaceElongatedWords(text);
    text = removeRepeatedPunctuation(text);
    text = removeExtraWhitespace(text);
    text = removeURLShorteners(text);
    text = removeSpacesTweets(text);
    text = removeShortTweets(text);
    return text;