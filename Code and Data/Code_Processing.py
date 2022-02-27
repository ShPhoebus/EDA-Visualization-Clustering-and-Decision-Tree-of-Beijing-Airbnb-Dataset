#-*-coding:GBK -*-
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as  pd
from sklearn.decomposition import  PCA
# from sklearn.preprocessing import Imputer
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder,OneHotEncoder##���ȱ���
# from scipy.interpolate import spline
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import  LinearRegression
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
import wordcloud
from sklearn.metrics import silhouette_score
import jieba
from matplotlib.dates import AutoDateLocator,DateFormatter
#ԭ�汾 from pyecharts.charts import Map,Geo
from pyecharts import Map,Geo#Geo�������ֲ�ͼ pip install pyecharts==0.1.9.4
from xpinyin import Pinyin

def area_score():
    data=pd.read_csv('listings11.csv')
    #data = pd.read_csv('listings11.csv').dropna(axis=0, how='any') #����һ��ʼ�Ƴ��������޷���ƽ��ֵ
    
    libs=[]#Ϊ��������avg_price��׼��������Ԫ����ȥ������ת��Ϊfloat������
    for lib in data['price']:
        libs.append(lib.split('$')[1].replace(',','').strip())#��price��Ķ���ȥ����
    data['price']=[float(m) for m in libs]
    area=set(data['neighbourhood_cleansed'].tolist())   #��������߾���ת�����б�
    area_score=[]#���������������÷֣��������������۵÷�����ͼ
    kmeans_score=[]#��������ָ������������÷֣�����������
    avg_price=[]#����ƽ��ֵ
    for i in area:#�ֱ�������������б�ֵ
        # print((i,sum(data['neighbourhood_cleansed']==i)))

        #���ĵ������ֱ�pinyin
        pinname = Pinyin()    #p.get_pinyin(u"�Ϻ�", ' ')
        pinname2 = pinname.get_pinyin(i.split('/')[0].strip(), '')
        pinname2 = pinname2[:-2]    #ȥ��qu

        kmeans_score.append((pinname2,data.loc[data['neighbourhood_cleansed']==i,'review_scores_rating':'review_scores_value'].sum().sum()/sum(data['neighbourhood_cleansed']==i)))
        area_score.append((i,data.loc[data['neighbourhood_cleansed']==i,'review_scores_location'].sum()/sum(data['neighbourhood_cleansed']==i)))
        avg_price.append((data.loc[data['neighbourhood_cleansed']==i,'price'].sum()/sum(data['neighbourhood_cleansed']==i)))
    # print(avg_price)
    pure_area=[]#������
    score=[]#����
    for j in area_score:#�������ͷ����ֿ���������������ͼ����

        #���ĵ������ֱ�pinyin,�˴�����
        #pinname = Pinyin()    #p.get_pinyin(u"�Ϻ�", ' ')
        #pinname2 = pinname.get_pinyin(j[0].split('/')[0].strip(), '')
        #pinname2 = pinname2[:-2] #ȥ��qu
        #pure_area.append(pinname2)
        
        pure_area.append(j[0].split('/')[0].strip())
        score.append(j[1])

    #location review score��ͼ����ͼ
    map2 =Map("Beijing location review score map", 'Beijing', width=1200, height=600)
    city,values2 = pure_area,score
    map2.add('Beijing', city, values2, visual_range=[1, 10], maptype='����', is_visualmap=True, visual_text_color='#000')
    map2.render(path="Beijing location review score map.html")

    #average price��ͼ����ͼ
    map3=Map('Beijing average price map','Beijing',width=1200,height=600)
    city3,values3=pure_area,avg_price
    map3.add('Beijing', city3, values3, visual_range=[1, 2000], maptype='����', is_visualmap=True, visual_text_color='#000')
    map3.render(path="Beijing average price map.html")

    '''#���� ����Ϊʲô����ʱ�۸����Ǻ���
    plt.rcParams['font.sans-serif'] = 'SimHei'#��ʾ����
    kmeans_data=pd.DataFrame(kmeans_score)
    kmeans_data.columns=['area','average_price']
    # print(kmeans_data)
    color=['red','blue','green','black','pink','purple','yellow']
    silhouette_score=[]
    for i in range(2,9):#ּ���ҳ���Ѵ���ƽ���ͣ�����Ѿ���Ч��
        model=KMeans(random_state=0,n_clusters=i)
        model.fit_transform(kmeans_data['average_price'].values.reshape(-1,1))
        silhouette_score.append(model.inertia_)#�ҳ���Ѵ���ƽ���ͣ������ŵľ�����Ŀ #estimator.inertia_����������ľ�ֵ�������ܺ�
    plt.plot(range(2,9),silhouette_score,'o-')
    plt.xlabel('K value')
    plt.ylabel('Sum of squares of clustering') #����ƽ���� ԽСԽ��
    plt.show()
    best_score=silhouette_score.index(min(silhouette_score))+1
    # print('��Ѿ�������Ϊ��',best_score)
    n_cluster=4 #��Ȼ��ã�������ֻѡ4��
    estimator=KMeans(random_state=0,n_clusters=4)
    y=estimator.fit_predict(kmeans_data['average_price'].values.reshape(-1,1)) #reshapepΪ��Ϊһ��
    # print(y)
    plt.figure(figsize=(9,6))   #ָ��figure�Ŀ�͸ߣ���λΪӢ��
    for i in range(len(y)):
        if y[i]==0:
            plt.scatter(kmeans_data.iloc[i,0],kmeans_data.iloc[i,1],marker='x',c='red')
        elif y[i]==1:
            plt.scatter(kmeans_data.iloc[i, 0], kmeans_data.iloc[i, 1], marker='o', c='blue')
        elif y[i]==2:
            plt.scatter(kmeans_data.iloc[i, 0], kmeans_data.iloc[i, 1], marker='.', c='green')
        else:
            plt.scatter(kmeans_data.iloc[i, 0], kmeans_data.iloc[i, 1], marker='*', c='purple')
    plt.xlabel('area')
    plt.ylabel('average price��RMB��')
    plt.show()'''


    return data

''' �����⣬����һ����ͬ
    # print(y)
    # fig=plt.figure(figsize=(12,18))
    fig=plt.figure(figsize=(9,6))
    plt.rcParams['font.sans-serif']='SimHei'
    plt.scatter(kmeans_data['area'],kmeans_data['average_price'],marker='o',c='green')
    plt.xlabel('area')
    plt.ylabel('average price��RMB��')
    plt.show()
    plt.figure(figsize=(9,6))
'''

def Mat_plot(data):
    room_counts={}
    for i in data['property_type'].tolist():
        # print(i,(data['property_type']==i).sum())#�˴���ʾÿ�����͵������ֱ��ж���
        room_counts[i]=room_counts.get(i,0)+1###########ͳ��ÿ�ַ��͵�����
    items=list(room_counts.items())
    items.sort(key=lambda x:x[1],reverse=True)#�Է���������������
    for non in items[5:]:#########################Ҫ�ּ���
        data.loc[data['property_type']==non[0],'property_type']='Other'#ת����ֻ�����ַ���{'Other', 'Serviced apartment', 'Loft', 'Condominium', 'House', 'Apartment'}
    room_num=[]
    for room in set(data['property_type'].tolist()):
        room_num.append((room,(data['property_type']==room).sum()))###[('Apartment', 12815), ('Other', 6989), ('Condominium', 5815), ('Loft', 2555), ('House', 4650), ('Serviced apartment', 1920)]
    room_=pd.DataFrame(room_num)
    room_.columns=['type','number']
###########################��ÿ�����͵ķ��ͻ���״ͼ#####################################
    fig=plt.figure(figsize=(12,6))
    ax=plt.subplot()
    ax.bar(room_['type'],room_['number'],color='lightblue')
    plt.show()




def plot_scatter(df,x,y,z,h):######����һ����ͼ�ĺ����������ڻ�ÿһ����·ݱ仯ʱ����
    plt.plot_date(df.date, df.eval(h), fmt='b.')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=24))
    plt.xticks(rotation=90, ha='center')
    label = ['{}_counts'.format(y)]
    plt.legend(label, loc='upper right')
    plt.grid()
    ax.set_title(u'{}_counts_monthly'.format(y), fontproperties='SimHei', fontsize=14)
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(z))

#���п�ʼ�����ߣ���Ч������   
#    begin,end=min(df.date),max(df.date)
#    day=[]
#    for i in range((end - begin).days + 1):
#        day.append(begin + datetime.timedelta(days=i))
#    plt.plot(day,df.eval(h))

#������
    # model=LinearRegression()
    # model.fit(df['price'].values.reshape(-1,1),df.eval(h))
    # # print(model.intercept_)
    # x_new=day
    # # pd.plotting.register_matplotlib_converters()
    # y_new=model.predict(df['price'].values.reshape(-1,1))
    # plt.plot(x_new,y_new,linewidth=2)

    plt.show()


########################������������״ͼ#########################################
def review(data,dataset):
    date_comments=[]
    dataset['time']=pd.to_datetime(dataset['date'])
    for date in set(dataset['time'].tolist()):
        date_comments.append((date,(dataset['time']==date).sum()))##ÿ����������ͳ��
    comments_counts=pd.DataFrame(date_comments)
    comments_counts.columns=['date','counts']
    comments_counts.to_csv('comments.csv') #�����ڻ�õ�·����
    df=pd.read_csv('comments.csv',parse_dates=['date'])
    plt.plot_date(df.date,df.counts,fmt='b.')   #��������ͼ
    ax=plt.gca()    #��ǰ��ͼ�����ͼ����ʹ��plt.gcf()��plt.gca()��ã��ֱ��ʾGet Current Figure��Get Current Axes����pyplotģ���У���ຯ�����ǶԵ�ǰ��Figure��Axes������д�������˵��plt.plot()ʵ���ϻ�ͨ��plt.gca()��õ�ǰ��Axes����ax��Ȼ���ٵ���ax.plot()����ʵ�������Ļ�ͼ��
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))   ###�������̶ȱ�ǩ��λ��,��ǩ�ı��ĸ�ʽ
    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=24))
    plt.xticks(rotation=90, ha='center') #plt.xticks([0,1],[1,2],rotation=0) [0,1]����x�������0��1λ�ã�[2,3]����0,1λ�õ���ʾlable��rotation����lable��ʾ����ת�Ƕ�
    label = ['comments_counts']
    plt.legend(label, loc='upper right')    #plt.legend����������Ҫ�����þ��Ǹ�ͼ����ͼ����plt.legend([x,y,z])����Ĳ���ʹ�õ���list�ĵ���ʽ��ͼ��ĵ����Ƹ��⺯��
    plt.grid()  ##���������ã�plt.grid()��
    ax.set_title(u'comments_counts_yearly', fontproperties='SimHei', fontsize=14)
    ax.set_xlabel('year')
    ax.set_ylabel('number')
    plt.show()  #����������ʱ��ı�
    df['price']=data['price']
    data_date=df.copy()
    data_review=data.copy()
    year=[[],[],[]]
    for y in data_date['date'].tolist():
        for z in range(len(year)):
            year[z].append(str(y).split('-')[z].strip())
    data_date['year'],data_date['month'],data_date['day']=year[0],year[1],year[2]

    ###########������ʾÿ�յ�������###############################

    # for uniq_year in set(data_date['year'].tolist()):##���Ҫ��ʾȫ����ݣ�������ȡ��ע�ͣ�������һ��ע�͵�����
    for uniq_year in ['2017','2018','2019']:
        data_date.loc[data_date['year']==uniq_year,:].to_csv('{}_comments.csv'.format(uniq_year))#��ÿһ��Ĳ�ͬ�·ݵ�comments�ֱ𱣴�Ϊcsv�ļ�  #.format�ַ�����ʽ��
        dataframe=pd.read_csv('{}_comments.csv'.format(uniq_year),parse_dates=['date'])
        plot_scatter(dataframe,uniq_year,'comments','number','counts')  #����ÿ������۱仯ͼ��

#########################������ʾÿ�յļ۸�仯##################

    for uniq_price_year in ['2017','2018']:
        dataframe = pd.read_csv('{}_comments.csv'.format(uniq_price_year), parse_dates=['date'])
        plot_scatter(dataframe, uniq_price_year, 'price', 'price', 'price') #����ÿ��ļ۸�仯ͼ��

##########################SuperHost##################
    number=[]
    occupancy_data=pd.concat([data['host_response_rate'],data['host_is_superhost'],data['review_scores_rating']],axis=1)
    for numb in occupancy_data['host_response_rate']:
        number.append(str(numb).split('%')[0].strip())
    occupancy_data['host_response_rate']=number

    # print(occupancy_data.isnull().sum())
    occupancy_data2=occupancy_data.dropna(axis=0,how='any')
    occupancy_data2=occupancy_data2[~occupancy_data2['host_response_rate'].isin(['nan'])]######����û�п�ֵ������nanֵ��ɾ�� #isin()����һ���б��жϸ�����Ԫ���Ƿ����б��У����ķ�����������ǰ����� ~

    superhost_y=occupancy_data2.loc[occupancy_data2['host_is_superhost']=='t',:]
    superhost_f=occupancy_data2.loc[occupancy_data2['host_is_superhost']=='f',:]
    plt.figure()
    plt.scatter(superhost_y['host_response_rate'].astype(int),superhost_y['review_scores_rating'],c='red',label='SuperHost:True',marker='.',alpha=0.5)  #astype()    ���������ͽ���ת��   
    plt.scatter(superhost_f['host_response_rate'].astype(int),superhost_f['review_scores_rating'],c='blue',label='SuperHost:False',marker='.',alpha=0.5)
    plt.grid(True)
    plt.grid(color='gray')
    plt.grid(linestyle='--')
    plt.xlabel('Host Response Rate')
    plt.ylabel('Avg Rating')
    plt.legend(loc=2)   #ͼ��λ��
    plt.show()



def box_plot():
    pass

###################################word_cloud��������#####################################
def word_cloud(data):
    word_data=data.copy()
    word_comments=word_data['comments'].tolist()
    chinese_word=[]
    english_word=[]
    for word in word_comments:
        try:
            english_word.append(''.join(byte for byte in word if ord(byte)<256 and byte not in ",!@#$|%^&*()<>?:{}"))##Ӣ���ı�
            chinese_word.append(''.join(byte.strip() for byte in word.strip() if ord(byte)>256 and byte not in "''!@#$%^&*()_+"))####�����ı�
        except:
            pass
    ######Ӣ�Ĵ���
    w=wordcloud.WordCloud(background_color='white',width=1000,height=700)
    w.generate(str(english_word))
    w.to_file('english_wordcloud.jpg')###����ΪjpgͼƬ
    ######���Ĵ���
    ls_chinese=jieba.lcut(str(chinese_word))
    chinese_text=''.join(ls_chinese)
    w2=wordcloud.WordCloud(font_path='simsun.ttc',width=1000,height=700,background_color='white')
    w2.generate(chinese_text)
    w2.to_file('chinese_wordcloud.jpg')

############################������Ԥ��#########################
def Decision_Tree():
    mix_data=pd.read_csv('decisiontree.csv')####��ȡ����

    ############################################����Ԥ����##########################################
    mix_data=mix_data[mix_data['host_is_superhost'].notna()]#host_is_superhostֻ��һ��Ϊ�գ�ɾ�� ##.notna()���������������ȱʧֵ
    mix_data['bathrooms']=mix_data['bathrooms'].fillna(mix_data['bathrooms'].mean())#��bathroomsȱʧֵ�þ�ֵ���
    mix_data['bedrooms']=mix_data['bedrooms'].fillna(mix_data['bedrooms'].mean())#��bedroomsȱʧֵ�þ�ֵ���
    mix_data['beds'] = mix_data['beds'].fillna(mix_data['beds'].mean())#��bedsȱʧֵ�þ�ֵ���
    security_deposit=[]
    for security in mix_data.loc[mix_data['security_deposit'].notnull(),'security_deposit'].tolist():##��security_depositָ��ǿյĲ��ִ�����Ԫ���ŵ���ֵȥ����Ԫ���ţ�������Ϊfloat����ֵ
        security_deposit.append(security.split('$')[1].replace(',','').strip())#�������м�Ķ���ȥ����1,456->1456)
    mix_data.loc[mix_data['security_deposit'].notnull(),'security_deposit']=[float(t) for t in security_deposit]###������ֵ���ͣ�ԭ��Ϊ�ַ�������Ϊfloat
    mix_data['security_deposit']=mix_data['security_deposit'].fillna(mix_data['security_deposit'].mean())#�ô����Ϊ��ֵ�͵����ֵ�ƽ��ֵ����ֵ
    cleaning_fee=[]#############cleaning_feeָ��Ĵ�����ͬ��
    for cleaning in mix_data.loc[mix_data['cleaning_fee'].notnull(),'cleaning_fee'].tolist():
        cleaning_fee.append(cleaning.split('$')[1].replace(',','').strip())
    mix_data.loc[mix_data['cleaning_fee'].notnull(),'cleaning_fee']=[float(t) for t in cleaning_fee]
    mix_data['cleaning_fee']=mix_data['cleaning_fee'].fillna(mix_data['cleaning_fee'].mean())
    # print(cleaning_fee)
    # mix_data['security_deposit'] = mix_data['security_deposit'].fillna(mix_data['security_deposit'].astype(float).mean())
    for nadata in ['review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_value','review_scores_location','reviews_per_month']:
        mix_data[nadata]=mix_data[nadata].fillna(mix_data[nadata].mean())#############������ָ��Ŀ�ֵ�ֱ��þ�ֵ���
        # print('����{}��������'.format(nadata))
    mix_data=mix_data[~mix_data['host_response_rate'].isnull()]#####��host_response_time����Ϊ��ֵ��ɾ����ֻɾ��4000���ң��Խ��Ӱ�첻��
    mix_data.index=range(mix_data.shape[0])
    ########################################���˿�ֵ��ȫ��������#############################

    # print(mix_data.isnull().sum())
    # print(mix_data.shape)
    mix_data.drop_duplicates(inplace=True)###ȥ���ظ�ֵ

    list_=list(mix_data['host_is_superhost'])
    for i,j in enumerate(list_):####################��host_is_superhostָ���t��Ϊ1��f��Ϊ0
        if j=='t':
            list_[i]=1
        else:
            list_[i]=0
    mix_data['host_is_superhost']=list_

    ##############��neighbourhood_cleansed�����Ʊ�������##############################

        
    mix_data['neighbourhood_cleansed']=[neibour.split('/')[0].strip() for neibour in mix_data['neighbourhood_cleansed']]
    One_hot_encoder = OneHotEncoder().fit(mix_data['neighbourhood_cleansed'].values.reshape(-1,1)) #OneHotEncoder �����ǳ�ʵ�ã�������ʵ�ֽ�����������ÿ��Ԫ��ת��Ϊһ���������������ֵ
    One_data=OneHotEncoder().fit_transform(mix_data['neighbourhood_cleansed'].values.reshape(-1,1)).toarray()
    neibourhood_cleansed=pd.DataFrame(One_data)
    neibourhood_cleansed.columns=One_hot_encoder.get_feature_names() #get_feature_names():����һ�������������Ƶ��б�ͨ�����������������one-hot��ʾ������������ʾ��Ӧ��������
    neibourhood_cleansed.index=range(neibourhood_cleansed.shape[0])
    mix_data=pd.concat([neibourhood_cleansed,mix_data.drop(['neighbourhood_cleansed'],axis=1)],axis=1)
    # print(neibourhood_cleansed)

    dataframe=[]
    features=['property_type','room_type','bed_type','cancellation_policy']
    for feature in features:
        one_hot_label=OneHotEncoder().fit(mix_data[feature].values.reshape(-1,1))
        one_hot_data=OneHotEncoder().fit_transform(mix_data[feature].values.reshape(-1,1)).toarray() #��listֱ��תΪObject[] ����
        feature_df=pd.DataFrame(one_hot_data)
        feature_df.columns=one_hot_label.get_feature_names()
        dataframe.append(feature_df)
        # print('{}deal done'.format(feature))
    mix_data=pd.concat([mix_data,pd.concat([dataframe[0],dataframe[1],dataframe[2],dataframe[3]],axis=1)],axis=1).drop(features,axis=1)
    list_2= list(mix_data['host_identity_verified'])
    for i, j in enumerate(list_):
        if j == 't':
            list_2[i] = 1
        else:
            list_2[i] = 0
    mix_data['host_identity_verified']=list_2
    mix_data['price']=[price.split('$')[1].replace(',','').strip() for price in mix_data['price']]
    mix_data['extra_people']=[people.split('$')[1].replace(',','').strip() for people in mix_data['extra_people']]
    mix_data['host_response_rate']=[rate.split('%')[0].replace(',','').strip() for rate in mix_data['host_response_rate']]
    mix_data['amenities']=[len(amenities) for amenities in mix_data['amenities']]
    mix_data[['price','host_response_rate']]=mix_data[['price','host_response_rate']].astype('float')
    # print(mix_data.shape)
    # print(mix_data.dtypes)
    ###########################################����ȫ������ת��Ϊ��ֵ��############################################

    y=pd.DataFrame(mix_data.loc[:,'price'])#############��ǩ��
    x=mix_data.drop(['price'],axis=1)###############������
    # print(x.info)
    x=x.astype('float')
    # y=y.astype('int')
    # print(x.dtypes)
    ###���������ݴ���----��׼��Z-scores####
    # x=(x-x.mean())/np.std(x)*100

    x.to_csv('decisiontree_dealdata.csv')###################���д���õ����� �����csv�ļ�
   #     # print(mix_data['host_is_superhost'])
    model=DecisionTreeRegressor(random_state=30,criterion='mse',max_depth=10) #random_stateָ�����������ʱ�����㷨��ʼ������ֵ
    data_split=int(x.shape[0]*0.7)
    # print(data_split)
    x_train,x_test,y_train,y_test=x.iloc[:data_split,:],x.iloc[data_split:,:],y.iloc[:data_split,:],y.iloc[data_split:,:]##���ֲ��Լ���ѵ��������Ϊ���ݼ�����û��˳������ȡǰ70%��Ϊѵ��������30%��Ϊ���Լ�
    x_test.index,y_test.index=range(x_test.shape[0]),range(y_test.shape[0])###�������� һ��Ҫ���ã���������
    model.fit(x_train,y_train)#ģ�͵�fit
    predict=model.predict(x_test)#�Բ��Լ���Ԥ��
    convent=pd.concat([pd.DataFrame(predict),y_test],axis=1)
    convent.columns=['prediction value','real value']
    #��άǰ�ľ���ƽ�����
    print('Mean Absolute Deviation without dimensionality reduction��', np.abs(convent['prediction value']-convent['real value']).mean())#��άǰ�ľ���ƽ�����
    # print(convent)
    convent.to_csv('δ��άʱ��Ԥ��Ƚ�.csv')######################################���������������̫�࣬Ԥ��Ч�����Ǻܺã���ά����
    plt.figure()
    plt.plot(range(y_test.shape[0]),predict,c='red',label='predict price')
    plt.plot(range(y_test.shape[0]),y_test,c='lightblue',label='real price')
    plt.title('compare with prediction without dimensionality reduction')
    plt.legend(loc=2)
    plt.show()

    ###��ά
    x_score=[]
    for i in range(1,20):###�ҵ���ѽ�ά���������
        pca_x=PCA(n_components=i,svd_solver='full').fit(x) #��ָ������ֵ�ֽ�SVD�ķ��������������ֽ�������ֵ�ֽ�SVD��һ��������һ���PCA�ⶼ�ǻ���SVDʵ�ֵġ���4������ѡ���ֵ��{��auto��, ��full��, ��arpack��, ��randomized��}��randomizedһ��������������������ά�ȶ�ͬʱ���ɷ���Ŀ�����ֽϵ͵�PCA��ά����ʹ����һЩ�ӿ�SVD������㷨�� full���Ǵ�ͳ�����ϵ�SVD��ʹ����scipy���Ӧ��ʵ�֡�arpack��randomized�����ó������ƣ�������randomizedʹ�õ���scikit-learn�Լ���SVDʵ�֣���arpackֱ��ʹ����scipy���sparse SVDʵ�֡�Ĭ����auto����PCA����Լ�ȥ��ǰ�潲���������㷨����ȥȨ�⣬ѡ��һ�����ʵ�SVD�㷨����ά��һ����˵��ʹ��Ĭ��ֵ�͹��ˡ�
        # print(pca_x.explained_variance_ratio_)
        x_score.append(pca_x.explained_variance_ratio_.sum())#explained_variance_ratio_������ά��ĸ����ɷֵķ���ֵռ�ܷ���ֵ�ı������������Խ����Խ����Ҫ�����ɷ֡�   #explained_variance_ratio_������ÿ������������ʣ������ܺ�Ϊ1��explained_variance_Ϊ����ֵ��ͨ������ʹ���������������Ի����������ͼ���߷���ֵͼ�����ڹ۲�PCA��ά���ֵ
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.figure()
    plt.plot(range(len(x_score)),x_score,c='red',label='trend of explained variance ratio with changing number of attribute',linewidth=4) #������������潵ά���������������
    plt.title('trend graph of explained variance ratio with changing number of attribute')
    plt.legend(loc=4)
    plt.show()
    ####�۲���ͼ��󣬷�����8���ʱ����ڱ仯����������ѡ���Ϊ8��###############
    pca_best=PCA(n_components=8,svd_solver='full').fit_transform(x)#��ά��Ϊ8����������
    x_best=pd.DataFrame(pca_best)
    x_best_train,x_best_test,y_best_train,y_best_test=x_best.iloc[:data_split,:],x_best.iloc[data_split:,:],y.iloc[:data_split,:],y.iloc[data_split:,:]
    x_best_test.index,y_best_test.index=range(x_test.shape[0]),range(y_test.shape[0])
    model.fit(x_best_train,y_best_train)
    best_predict=model.predict(x_best_test)
    best_convent=pd.concat([pd.DataFrame(best_predict),pd.DataFrame(y_best_test)],axis=1)
    best_convent.columns=['prediction value','real value']
    #��ά��ľ���ƽ�����
    print('Mean Absolute Deviation with dimensionality reduction��', np.abs(best_convent['prediction value'] - best_convent['real value']).mean())
    best_convent.to_csv('��ά���Ԥ��Ƚ�.csv')
    plt.figure()
    plt.plot(range(best_predict.shape[0]),best_predict,c='red',label='prediction value')
    plt.plot(range(y_best_test.shape[0]),y_best_test,c='lightblue',label='real value')
    plt.title('compare with prediction with dimensionality reduction')
    plt.legend(loc=2)
    plt.show()


def K_means(data):###����
    k_data=data[['accommodates','price','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','beds','bedrooms']]
    # print(k_data)
    #���ݴ���
    k_data=k_data.fillna(k_data.mean())##�þ�ֵ����ֵ
    #�������ı�ƴ��
    pinname = Pinyin()
    k_data['area'] = [pinname.get_pinyin(area.split('/')[0].strip(), '')[:-2] for area in data['neighbourhood_cleansed']]
    x=k_data.drop(['price'],axis=1)#�������󣬽�����һ��ȥ��
    # print(x.isnull().sum())
    y=k_data.loc[:,'price']
    silh_score=[]#����ϵ����������ϵ�����ж���ѵľ�������������ϵ��Խ��Խ�ţ������ŵĴ�����kֵ��������ϵ����ֵ�ǽ��� [-1,1] ��Խ������1�����ھ۶Ⱥͷ���ȶ���Խ���

    color = ['red', 'blue', 'green', 'black', 'pink', 'purple', 'yellow']
    for i in range(2,7):#�ҵ���Ѿ����kֵ����ѧϰ���ߵķ�ʽ��������������Ϊ2��
        model=KMeans(n_clusters=i).fit(k_data.drop(['area'],axis=1))#ģ�͵�fit
        y_predict=model.predict(k_data.drop(['area'],axis=1))#ѵ��ģ�ͺ�ľ�����
        silh_score.append(silhouette_score(x.drop(['area'],axis=1),y_predict))###����ͬkֵ�µ�����ϵ��
        print('done number {}'.format(i))##############���������е���һ����
        plt.figure(figsize=(9,6))
        for j in range(i):
            k_data['label']=model.labels_
            plt.scatter(k_data.loc[k_data['label']==j]['area'],k_data.loc[k_data['label']==j]['price'],marker='+',c=color[j])
        plt.title('the clustering effect when k value is {}'.format(i))
        plt.xlabel('area')
        plt.ylabel('price')
        plt.show()


    best_score=silh_score.index(max(silh_score))+2#����������ϵ���µ�kֵ���࣬�����ӻ�
    print('Best number of clustering��',best_score)
    plt.figure()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.plot(range(2,len(silh_score)+2),silh_score,c='black')
    plt.title('Silhouette Coefficient trend with changing number of clustering')
    plt.show()
################����ѵĴ�����kֵ����ͼ
    model2=KMeans(n_clusters=best_score)
    model2.fit(k_data.drop(['area'],axis=1))
    k_data['label']=model.labels_#��Ԥ���ǩ�ӵ����ݼ���
    plt.figure(figsize=(9,6))
    color=['red','blue','green','black','pink','purple','yellow']
    for i in range(best_score):
        plt.scatter(k_data.loc[k_data['label']==i]['area'],k_data.loc[k_data['label']==i]['price'],marker='+',c=color[i])
    plt.xlabel('area')
    plt.ylabel('price')
    plt.title('best cluster according price')
    plt.show()
    # for i in range(best_score):


    k_data=k_data.fillna(k_data.mean())
    # print(k_data.info,k_data.isnull().sum())
    # print(k_data.shape)


if __name__=='__main__':
    data=area_score()
    dataset=pd.read_csv('reviews.csv')
    Mat_plot(data)
    review(data,dataset)
    word_cloud(dataset)
    Decision_Tree()
    K_means(data)

'''��������1������Ԥ����
            �������ݵ�ȥ�ء��������ֵ����ֵ���������þ�ֵ��䡣����ַ��͵���ֵ�����������������Ԫ���Ż��߰ٷֺţ�������ȥ����������ֵ�еĶ���ȥ������astype�ķ�������Ϊfloat
            ����ֵ�����ô���õ���ֵ�þ�ֵ����ֵ�����ַ����͵����������������host_is_superhost�ȣ��Ʊ�������
           2�����ֲ��Լ���ѵ��������Ϊ���ݼ����������ԣ�����ȡǰ70%Ϊѵ���������µ�Ϊ���Լ���
           3��ģ�͵Ľ���  4��ģ�ͶԲ��Լ���Ԥ�Ⲣ��ԭ���ݱȽϡ�
           
    k_means���ࣺ����Ҳ������Ԥ�����þ�ֵ����ֵ��Ϊ���ж���ѵľ����kֵ��������ϵ������������ϵ��Խ�󣬾���Ч��Խ�á�����������
                    ��ѧϰ���ߵķ������ҵ��������ϵ���µ�kֵ���������kֵ���࣬�����ӻ�����Ч��'''