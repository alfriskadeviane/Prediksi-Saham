import re
from flask import Flask, render_template, request, send_file
from numpy.lib import index_tricks
import os
import io
import json
import matplotlib.pyplot as plt
import mplfinance as fplt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from werkzeug.utils import send_file
app = Flask(__name__)
from model import *


data=loadData()
#data['Date']=pd.to_datetime(data['Date'])
#saham=data.set_index(pd.DatetimeIndex(data['Date']))
dataclear,dataNorm=normData(data)
inp=pd.DataFrame(dataNorm, columns=["Open","High","Low","Volume"])
out=pd.DataFrame(dataNorm, columns=["Close"])
Xtrain, Xtest, ytrain, ytest=splitData(dataNorm)
xt=pd.DataFrame(Xtrain, columns=["Open","High","Low","Volume"])
yt=pd.DataFrame(ytrain, columns=["Close"])
xts=pd.DataFrame(Xtest,columns=["Open","High","Low","Volume"])
yts=pd.DataFrame(ytest, columns=["Close"])
nn=NeuralNet(layers=[4,2,1], learning_rate=0.01, epoch=500)
w1,b1,w2,b2=nn.init_weights()
w1=pd.DataFrame(w1)
w2=pd.DataFrame(w2).T
b1=pd.DataFrame(b1).T
b2=pd.DataFrame(b2)
wb1,bb1,wb2,bb2=nn.fit(Xtrain,ytrain)
wb1=pd.DataFrame(wb1)
wb2=pd.DataFrame(wb2).T
bb1=pd.DataFrame(bb1).T
bb2=pd.DataFrame(bb2)
train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)
tp=pd.DataFrame(train_pred,columns=["trainpred"])
t=pd.DataFrame(ytrain,columns=["ytrain"])
dc=pd.concat([t,tp],axis=1)
# nmse=nn.mse(ytrain, train_pred)
nmse=mean_squared_error(ytrain, train_pred)
pop=nn.popawal(popsize=50)
dp=pd.DataFrame(pop)
wp1,bp1,wp2,bp2,output,wa1,ba1,wa2,ba2=nn.fitalgen(Xtrain,ytrain,popsize=50,cr=0.9 ,mr=0.5 ,gensize=10)
wp1=pd.DataFrame(wp1)
wp2=pd.DataFrame(wp2).T
bp1=pd.DataFrame(bp1).T
bp2=pd.DataFrame(bp2)
wa1=pd.DataFrame(wa1)
wa2=pd.DataFrame(wa2).T
ba1=pd.DataFrame(ba1).T
ba2=pd.DataFrame(ba2)
train_pred= nn.predict(Xtrain)
test_pred = nn.predict(Xtest)
to=pd.DataFrame(output,columns=["trainpred"])
do=pd.concat([t,to],axis=1)
gmse=nn.mse(ytrain, output)

@app.route("/")
def main():
    #close=json.dumps(close),date=json.dumps(date)
    return render_template('index.html', data=data.to_html(classes='table table-bordered table-striped table-hover'))

@app.route('/chart')
def chart():
    # fplt.plot(saham,type='candle',style='yahoo',savefig='static/assets/img/plot.png')
    figure=go.Figure(
    data= [
        go.Candlestick(
            #x=data['Date'],
            low=data['Low'],
            high=data['High'],
            close=data['Close'],
            open=data['Open']
            # increasing_line_color='green',
            # decreasing_line_color='red'
        )
    ]
    )
    figure.update_layout(
    title= 'BBCA Price',
    yaxis_title='BBCA Stock Price IDR',
    xaxis_title='Date'
    )
    graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    #print(data)
    #print(type(data))
    # fig=px.line(saham,x='Date',y="Close")
    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('chart.html',graphJSON=graphJSON)

@app.route("/normalisasi")
def normalisasi():
    return render_template('normalisasi.html',  dataNorm=dataNorm.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/split")
def split():
    return render_template('split.html',inp=inp.to_html(classes='table table-bordered table-striped table-hover'),out=out.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/training")
def training():
    return render_template('training.html',xt=xt.to_html(classes='table table-bordered table-striped table-hover'),yt=yt.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/testing")
def testing():
    return render_template('testing.html',xts=xts.to_html(classes='table table-bordered table-striped table-hover'),yts=yts.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/process")
def process():
    fig=px.line(dc,y=['ytrain','trainpred'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    figg=px.line(do,y=['ytrain','trainpred'])
    gJSON = json.dumps(figg, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('process.html',w1=w1.to_html(classes='table table-bordered table-striped table-hover'),w2=w2.to_html(classes='table table-bordered table-striped table-hover'),b1=b1.to_html(classes='table table-bordered table-striped table-hover'),b2=b2.to_html(classes='table table-bordered table-striped table-hover'),wb1=wb1.to_html(classes='table table-bordered table-striped table-hover'),wb2=wb2.to_html(classes='table table-bordered table-striped table-hover'),bb1=bb1.to_html(classes='table table-bordered table-striped table-hover'),bb2=bb2.to_html(classes='table table-bordered table-striped table-hover'), graphJSON=graphJSON, nmse=nmse, gmse=gmse,dp=dp.to_html(classes='table table-bordered table-striped table-hover'),wp1=wp1.to_html(classes='table table-bordered table-striped table-hover'),wp2=wp2.to_html(classes='table table-bordered table-striped table-hover'),bp1=bp1.to_html(classes='table table-bordered table-striped table-hover'),bp2=bp2.to_html(classes='table table-bordered table-striped table-hover'),wa1=wa1.to_html(classes='table table-bordered table-striped table-hover'),wa2=wa2.to_html(classes='table table-bordered table-striped table-hover'),ba1=ba1.to_html(classes='table table-bordered table-striped table-hover'),ba2=ba2.to_html(classes='table table-bordered table-striped table-hover'),gJSON=gJSON)
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method =='POST':
        open=float(request.form['open'])
        high=float(request.form['high'])
        low=float(request.form['low'])                                                        
        volume=float(request.form['volume'])
        q=[open, high, low, volume]
        q=pd.DataFrame(q).T
        q.columns=['Open', 'High', 'Low', 'Volume']
        d=(q-dataclear.min())/(dataclear.max()-dataclear.min())
        a=np.array(d.drop(columns=['Close']))
        test = nn.predict(a)
        hasil= test*(dataclear['Close'].max()-dataclear['Close'].min())+dataclear['Close'].min()
        
        if volume==0 or high==0 or open==0 or low==0:
            hasil='Pasar Saham Libur'
        elif volume<0 or high<0 or open<0 or low<0:
            hasil='Harga tidak mungkin bernilai negatif'
        elif high<low:
            hasil='High harus lebih besar dari low'
        elif open<low:
            hasil='Open harus lebih besar atau sama dengan low'
        elif open>high:
            hasil='Open harus lebih kecil atau sama dengan high'
        else:
            hasil=int(hasil.round())
        
        return render_template('predict.html',hasil=hasil)
    return render_template('predict.html')
@app.route("/prediction", methods=['GET','POST'])
def prediction():
    # Xtrain,ytrain=splitDataHarian(dataNorm)
    # nn=NeuralNet(layers=[4,2,1], learning_rate=0.01, epoch=500)
    # nn.fit(Xtrain,ytrain)
    # nn.fitalgen(Xtrain,ytrain,popsize=50,cr=0.9 ,mr=0.5 ,gensize=10)
    datenow=datetime.date.today()
    enddate=datenow+ datetime.timedelta(days=1)
    datas=yf.download("BBCA.JK", start=datenow, end=enddate)
    datasa= datas.replace(0, np.nan)
    datasaham=datasa.dropna()
    open=float(datasaham['Open'])
    high=float(datasaham['High'])
    low=float(datasaham['Low'])
    volume=int(datasaham['Volume'])
    close=float(datasaham['Close'])
    q=[open, high, low, volume, close]
    q=pd.DataFrame(q).T
    q.columns=['Open', 'High', 'Low', 'Volume', 'Close']
    d=(q-dataclear.min())/(dataclear.max()-dataclear.min())
    a=np.array(d.drop(columns=['Close']))
    test = nn.predict(a)
    c=np.array(d['Close'])
    hasil= test*(dataclear['Close'].max()-dataclear['Close'].min())+dataclear['Close'].min()
    hasil=int(hasil.round())
    mse=mean_squared_error(c, test)
    return render_template('prediction.html', open=open, high=high, low=low, volume=volume, hasil=hasil, mse=mse)
if __name__ == "__main__":
    app.run(debug=True)

