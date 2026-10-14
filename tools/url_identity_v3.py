"""Canonical identity using only supplied labels, excluding ambiguous groups."""
import html
from urllib.parse import urlsplit,unquote,parse_qsl,urlencode
import pandas as pd

def canonical(u):
 try:
  p=urlsplit(html.unescape(u if '://' in u else 'http://'+u))
  host=(p.hostname or '').lower().removeprefix('www.')
  path=unquote(p.path).rstrip('/') or '/'
  return host+':'+str(p.port or '')+path+'?'+urlencode(sorted(parse_qsl(p.query,keep_blank_values=True)))
 except ValueError:return u

def lookup(train,test):
 z=pd.DataFrame({'k':train.url.map(canonical),'y':train.result}).groupby('k').y.agg(['nunique','first'])
 return test.url.map(canonical).map(z[z['nunique'].eq(1)]['first'])
