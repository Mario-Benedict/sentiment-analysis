from math import ceil
from flask import Flask, flash, redirect, render_template, request, url_for
from utils.helper import check_file_exists, is_valid_file
from werkzeug.utils import secure_filename
import polars as pl
import os

app = Flask(__name__, template_folder='views')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = "5WABiFG8ColB1+kZmDZSmg1/I3RByYwfJi0QlMh2CkBXisw8mFXeoIgXONDiJ0Z2mRC76vY7P5gkVsc26oaXUw=="

@app.route('/dataset', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def home():
  df = None
  page = request.args.get('page', 1, type=int)
  per_page = request.args.get('per_page', 10, type=int)
  total_page = 0

  if check_file_exists(app.config['UPLOAD_FOLDER'], 'dataset.csv'):
    df = pl.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv'))
    df = df.rename({'Date': 'date', 'User': 'user', 'Review': 'review', 'Rating ': 'rating'})
    df = df.drop('Number ')

    total_page = ceil(df.shape[0] / per_page)

  if page > total_page:
    return redirect(f'dataset?page={total_page}', code=302)

  if page < 1:
    return redirect('dataset?page=1', code=302)

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
                          title='Home',
                          df=df.to_dicts() if df is not None else None,
                          columns=df.columns if df is not None else None,
                          rows=df.shape[0] if df is not None else None,
                          page=page,
                          per_page=per_page,
                          total_page=total_page,
                          pages=pages
                        )

@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing.html', title='Preprocessing')

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
