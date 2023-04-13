import os
import nltk
import demoji
import polars as pl
from textblob import TextBlob

from math import ceil
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from dotenv import dotenv_values
from werkzeug.utils import secure_filename
from utils.helper import check_file_exists, is_valid_file
from flask import Flask, flash, redirect, render_template, request, url_for
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords')
nltk.download('punkt')

secret_key = dotenv_values('.env').get('SECRET_KEY')

app = Flask(__name__, template_folder='views')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = secret_key

PER_PAGE = 10

stop_factory = StopWordRemoverFactory().get_stop_words()
stop_words = stopwords.words('indonesian')
stop_words.extend(stopwords.words('english'))
stop_words.extend([
  'yg', 'ga', 'gak', 'aja', 'ni',
  'nih', 'sih', 'kayak', 'kayanya',
  'ny', 'jd', 'jg', 'tp', 'tapi', 'klo',
  'jdi', 'jgn', 'jgk', 'jgkn', 'jgknya',
  'e', 'emng', 'emg', 'emgnya', 'emgkn',
  'krn', 'krna', 'utk', 'va', 'yu', 'yuk',
])
stop_words.extend(stop_factory)

happy_emoticons = [
  ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
  ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
  '=-3', '=3', 'B^D', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P',
  ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)',
  '>;)', '>:-)', '<3'
]

sad_emoticons = [
  ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<', ':-[', ':[', ':{', ':-||',
  '>:[', ':{', ':@', ":'-(", ":'(", 'D:<', 'D:', 'D8', 'D;', 'D=', 'DX',
  'v.v', "D-':", '>:(', '>:[', ':-/', '>.<', ':/', ':-.', '>.<'
]

stop_words.extend(happy_emoticons)
stop_words.extend(sad_emoticons)

stemmer = StemmerFactory().create_stemmer()

def get_df():
  df = None
  total_page = 0
  if check_file_exists(app.config['UPLOAD_FOLDER'], 'dataset.csv'):
    df = pl.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv'))
    df = df.rename({'Date': 'date', 'User': 'user', 'Review': 'review', 'Rating ': 'rating'})
    df = df.drop('Number ')

    total_page = ceil(df.shape[0] / PER_PAGE)

  return df, total_page

def get_pagination(page, total_page):
  page_start = 1
  page_to_show = 3
  pages = []

  while page_start <= total_page:
    if page_start <= page_to_show or page_start >= total_page - page_to_show + 1 or (page_start >= page - 1 and page_start <= page + 1):
      pages.append(page_start)
      page_start += 1

      continue

    pages.append('...')

    if page_start < page:
      page_start = page - 1
    else:
      page_start = total_page - page_to_show + 1

  return pages

@app.route('/dataset', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def home():
  df, total_page = get_df()

  page = request.args.get('page', 1, type=int)

  if page > total_page and total_page > 0:
    return redirect(f'dataset?page={total_page}', code=302)

  if page < 1:
    return redirect('dataset?page=1', code=302)

  pages = get_pagination(page, total_page)

  if request.method == 'POST':
    if 'dataset' not in request.files:
      flash('No file part', category='error')
      return redirect(request.url)

    file_upload = request.files['dataset']

    if file_upload.filename == '':
      flash('No selected file', category='error')
      return redirect(request.url)

    if file_upload and is_valid_file(file_upload.filename):
      file_upload.filename = 'dataset.csv'
      filename = secure_filename(file_upload.filename)
      file_upload.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], filename))

      flash('File uploaded successfully', category='success')
      return redirect(request.url, code=302)

    flash('Invalid file extension', category='error')
    return redirect(request.url, code=302)

  return render_template('index.html',
                          title='Sentiment Analysis',
                          df=df.to_dicts() if df is not None else None,
                          columns=df.columns if df is not None else None,
                          rows=df.shape[0] if df is not None else None,
                          page=page,
                          per_page=PER_PAGE,
                          total_page=total_page,
                          pages=pages,
                          url='dataset'
                        )

@app.route('/preprocessing')
def preprocessing():
  if check_file_exists(app.config['UPLOAD_FOLDER'], 'dataset.csv'):
    df, total_page = get_df()

    page = request.args.get('page', 1, type=int)

    if page > total_page and total_page > 0:
      return redirect(f'preprocessing?page={total_page}', code=302)

    if page < 1:
      return redirect('preprocessing?page=1', code=302)

    pages = get_pagination(page, total_page)

    # case folding the review column
    clean_df = df.with_columns([
      pl.col('review').str.to_lowercase()
    ])


    # remove unnecessary characters (e.g emojis, url, mention, and hashtag)
    clean_df = clean_df.with_columns([
      pl.col('review').str.replace(r'http\S+|@\S+|#\S+|', ' ').apply(lambda x: demoji.replace(x, ' ')),
    ])

    # reshape new dataframe
    clean_df = pl.DataFrame({
      'review': clean_df['review'],
      'total_words': clean_df['review'].apply(lambda x: len(x.split())),
      'total_characters': clean_df['review'].str.lengths(),
      'words_rate': clean_df['review'].apply(lambda x: round(len(x) / len(x.split()), 2)),
    })

    # remove punctuation
    punctuation_df = pl.DataFrame({
      'review': clean_df['review'].str.replace_all(r'[^\w\s]', ' ')
    })

    # add new column to punctuation_df
    punctuation_df = punctuation_df.with_columns([
      pl.lit(punctuation_df['review'].apply(lambda x: len(x.split()))).alias('total_words'),
      pl.lit(punctuation_df['review'].str.lengths()).alias('total_characters'),
      pl.lit(punctuation_df['review'].apply(lambda x: round(len(x) / len(x.split()), 2))).alias('words_rate'),
    ])

    tokenized_df = pl.DataFrame({
      'tokenized': punctuation_df['review'].apply(lambda x: word_tokenize(x))
    })

    # remove stop words
    stop_words_df = pl.DataFrame({
      'review': tokenized_df['tokenized'].apply(lambda x: ' '.join([word for word in x if word not in stop_words])),
      'stop_words_removed': tokenized_df['tokenized'].apply(lambda x: len([word for word in x if word not in stop_words]))
    })

    stop_words_df = stop_words_df.with_columns([
      pl.lit(stop_words_df['review'].apply(lambda x: len(x.split()))).alias('total_words'),
      pl.lit(stop_words_df['review'].str.lengths()).alias('total_characters'),
      pl.lit(stop_words_df['review'].apply(lambda x: round(len(x) / len(x.split()), 2))).alias('words_rate'),
    ])

    stemmed_df = pl.DataFrame({
      'review': stop_words_df['review'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    })

    stemmed_df = stemmed_df.with_columns([
      pl.lit(stemmed_df['review'].apply(lambda x: len(x.split()))).alias('total_words'),
      pl.lit(stemmed_df['review'].str.lengths()).alias('total_characters'),
      pl.lit(stemmed_df['review'].apply(lambda x: round(len(x) / len(x.split()), 2))).alias('words_rate'),
    ])

    return render_template('preprocessing.html',
                            title='Preprocessing',
                            clean_df=clean_df.to_dicts(),
                            clean_df_columns=clean_df.columns,
                            punctuation_df=punctuation_df.to_dicts(),
                            punctuation_df_columns=punctuation_df.columns,
                            stop_words_df=stop_words_df.to_dicts(),
                            stop_words_df_columns=stop_words_df.columns,
                            stemmed_df=stemmed_df.to_dicts(),
                            stemmed_df_columns=stemmed_df.columns,
                            rows=clean_df.shape[0],
                            page=page,
                            per_page=PER_PAGE,
                            total_page=total_page,
                            pages=pages,
                            url='preprocessing'
                          )

  return redirect(url_for('home'))

@app.route('/pembobotan')
def word_weighting():
  return render_template('word_weight.html', title='Pembobotan')


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
