#-*-coding:GBK -*-
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as  pd
from sklearn.decomposition import  PCA
# from sklearn.preprocessing import Imputer
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder,OneHotEncoder##独热变量
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
#原版本 from pyecharts.charts import Map,Geo
from pyecharts import Map,Geo#Geo是热力分布图 pip install pyecharts==0.1.9.4
from xpinyin import Pinyin

def area_score():
    data=pd.read_csv('listings11.csv')
    #data = pd.read_csv('listings11.csv').dropna(axis=0, how='any') #不能一开始移除，后面无法算平均值
    
    libs=[]#为后续计算avg_price做准备，将美元符号去除，并转换为float型数字
    for lib in data['price']:
        libs.append(lib.split('$')[1].replace(',','').strip())#将price里的逗号去除掉
    data['price']=[float(m) for m in libs]
    area=set(data['neighbourhood_cleansed'].tolist())   #将数组或者矩阵转换成列表
    area_score=[]#各个地区的量化得分，用于做地区评价得分热力图
    kmeans_score=[]#各个评价指标分数的量化得分，用于做聚类
    avg_price=[]#房价平均值
    for i in area:#分别计算以上三个列表值
        # print((i,sum(data['neighbourhood_cleansed']==i)))

        #中文地区名字变pinyin
        pinname = Pinyin()    #p.get_pinyin(u"上海", ' ')
        pinname2 = pinname.get_pinyin(i.split('/')[0].strip(), '')
        pinname2 = pinname2[:-2]    #去掉qu

        kmeans_score.append((pinname2,data.loc[data['neighbourhood_cleansed']==i,'review_scores_rating':'review_scores_value'].sum().sum()/sum(data['neighbourhood_cleansed']==i)))
        area_score.append((i,data.loc[data['neighbourhood_cleansed']==i,'review_scores_location'].sum()/sum(data['neighbourhood_cleansed']==i)))
        avg_price.append((data.loc[data['neighbourhood_cleansed']==i,'price'].sum()/sum(data['neighbourhood_cleansed']==i)))
    # print(avg_price)
    pure_area=[]#地区名
    score=[]#分数
    for j in area_score:#将地区和分数分开，方便以下热力图传参

        #中文地区名字变pinyin,此处出错
        #pinname = Pinyin()    #p.get_pinyin(u"上海", ' ')
        #pinname2 = pinname.get_pinyin(j[0].split('/')[0].strip(), '')
        #pinname2 = pinname2[:-2] #去掉qu
        #pure_area.append(pinname2)
        
        pure_area.append(j[0].split('/')[0].strip())
        score.append(j[1])

    #location review score地图热力图
    map2 =Map("Beijing location review score map", 'Beijing', width=1200, height=600)
    city,values2 = pure_area,score
    map2.add('Beijing', city, values2, visual_range=[1, 10], maptype='北京', is_visualmap=True, visual_text_color='#000')
    map2.render(path="Beijing location review score map.html")

    #average price地图热力图
    map3=Map('Beijing average price map','Beijing',width=1200,height=600)
    city3,values3=pure_area,avg_price
    map3.add('Beijing', city3, values3, visual_range=[1, 2000], maptype='北京', is_visualmap=True, visual_text_color='#000')
    map3.render(path="Beijing average price map.html")

    '''#聚类 存在为什么聚类时价格又是合理
    plt.rcParams['font.sans-serif'] = 'SimHei'#显示中文
    kmeans_data=pd.DataFrame(kmeans_score)
    kmeans_data.columns=['area','average_price']
    # print(kmeans_data)
    color=['red','blue','green','black','pink','purple','yellow']
    silhouette_score=[]
    for i in range(2,9):#旨在找出最佳簇内平方和，即最佳聚类效果
        model=KMeans(random_state=0,n_clusters=i)
        model.fit_transform(kmeans_data['average_price'].values.reshape(-1,1))
        silhouette_score.append(model.inertia_)#找出最佳簇内平方和，即最优的聚类数目 #estimator.inertia_代表聚类中心均值向量的总和
    plt.plot(range(2,9),silhouette_score,'o-')
    plt.xlabel('K value')
    plt.ylabel('Sum of squares of clustering') #簇内平方和 越小越优
    plt.show()
    best_score=silhouette_score.index(min(silhouette_score))+1
    # print('最佳聚类总数为：',best_score)
    n_cluster=4 #虽然多好，但这里只选4个
    estimator=KMeans(random_state=0,n_clusters=4)
    y=estimator.fit_predict(kmeans_data['average_price'].values.reshape(-1,1)) #reshapep为改为一列
    # print(y)
    plt.figure(figsize=(9,6))   #指定figure的宽和高，单位为英寸
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
    plt.ylabel('average price（RMB）')
    plt.show()'''


    return data

''' 有问题，与上一段相同
    # print(y)
    # fig=plt.figure(figsize=(12,18))
    fig=plt.figure(figsize=(9,6))
    plt.rcParams['font.sans-serif']='SimHei'
    plt.scatter(kmeans_data['area'],kmeans_data['average_price'],marker='o',c='green')
    plt.xlabel('area')
    plt.ylabel('average price（RMB）')
    plt.show()
    plt.figure(figsize=(9,6))
'''

def Mat_plot(data):
    room_counts={}
    for i in data['property_type'].tolist():
        # print(i,(data['property_type']==i).sum())#此处显示每种类型的数量分别有多少
        room_counts[i]=room_counts.get(i,0)+1###########统计每种房型的数量
    items=list(room_counts.items())
    items.sort(key=lambda x:x[1],reverse=True)#对房型数量进行排序
    for non in items[5:]:#########################要分几类
        data.loc[data['property_type']==non[0],'property_type']='Other'#转换成只有六种房型{'Other', 'Serviced apartment', 'Loft', 'Condominium', 'House', 'Apartment'}
    room_num=[]
    for room in set(data['property_type'].tolist()):
        room_num.append((room,(data['property_type']==room).sum()))###[('Apartment', 12815), ('Other', 6989), ('Condominium', 5815), ('Loft', 2555), ('House', 4650), ('Serviced apartment', 1920)]
    room_=pd.DataFrame(room_num)
    room_.columns=['type','number']
###########################对每种类型的房型画柱状图#####################################
    fig=plt.figure(figsize=(12,6))
    ax=plt.subplot()
    ax.bar(room_['type'],room_['number'],color='lightblue')
    plt.show()




def plot_scatter(df,x,y,z,h):######构造一个画图的函数，方便在画每一年的月份变化时调用
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

#此行开始画连线，但效果不佳   
#    begin,end=min(df.date),max(df.date)
#    day=[]
#    for i in range((end - begin).days + 1):
#        day.append(begin + datetime.timedelta(days=i))
#    plt.plot(day,df.eval(h))

#无作用
    # model=LinearRegression()
    # model.fit(df['price'].values.reshape(-1,1),df.eval(h))
    # # print(model.intercept_)
    # x_new=day
    # # pd.plotting.register_matplotlib_converters()
    # y_new=model.predict(df['price'].values.reshape(-1,1))
    # plt.plot(x_new,y_new,linewidth=2)

    plt.show()


########################评论数量的柱状图#########################################
def review(data,dataset):
    date_comments=[]
    dataset['time']=pd.to_datetime(dataset['date'])
    for date in set(dataset['time'].tolist()):
        date_comments.append((date,(dataset['time']==date).sum()))##每天评论数的统计
    comments_counts=pd.DataFrame(date_comments)
    comments_counts.columns=['date','counts']
    comments_counts.to_csv('comments.csv') #保存在获得的路径下
    df=pd.read_csv('comments.csv',parse_dates=['date'])
    plt.plot_date(df.date,df.counts,fmt='b.')   #日期序列图
    ax=plt.gca()    #当前的图表和子图可以使用plt.gcf()和plt.gca()获得，分别表示Get Current Figure和Get Current Axes。在pyplot模块中，许多函数都是对当前的Figure或Axes对象进行处理，比如说：plt.plot()实际上会通过plt.gca()获得当前的Axes对象ax，然后再调用ax.plot()方法实现真正的绘图。
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))   ###设置主刻度标签的位置,标签文本的格式
    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=24))
    plt.xticks(rotation=90, ha='center') #plt.xticks([0,1],[1,2],rotation=0) [0,1]代表x坐标轴的0和1位置，[2,3]代表0,1位置的显示lable，rotation代表lable显示的旋转角度
    label = ['comments_counts']
    plt.legend(label, loc='upper right')    #plt.legend（）函数主要的作用就是给图加上图例，plt.legend([x,y,z])里面的参数使用的是list的的形式将图表的的名称给这函数
    plt.grid()  ##网格线设置（plt.grid()）
    ax.set_title(u'comments_counts_yearly', fontproperties='SimHei', fontsize=14)
    ax.set_xlabel('year')
    ax.set_ylabel('number')
    plt.show()  #绘制评论随时间的表
    df['price']=data['price']
    data_date=df.copy()
    data_review=data.copy()
    year=[[],[],[]]
    for y in data_date['date'].tolist():
        for z in range(len(year)):
            year[z].append(str(y).split('-')[z].strip())
    data_date['year'],data_date['month'],data_date['day']=year[0],year[1],year[2]

    ###########按年显示每日的评论数###############################

    # for uniq_year in set(data_date['year'].tolist()):##如果要显示全部年份，在这里取消注释，并将下一行注释掉即可
    for uniq_year in ['2017','2018','2019']:
        data_date.loc[data_date['year']==uniq_year,:].to_csv('{}_comments.csv'.format(uniq_year))#将每一年的不同月份的comments分别保存为csv文件  #.format字符串格式化
        dataframe=pd.read_csv('{}_comments.csv'.format(uniq_year),parse_dates=['date'])
        plot_scatter(dataframe,uniq_year,'comments','number','counts')  #绘制每年的评论变化图表

#########################按年显示每日的价格变化##################

    for uniq_price_year in ['2017','2018']:
        dataframe = pd.read_csv('{}_comments.csv'.format(uniq_price_year), parse_dates=['date'])
        plot_scatter(dataframe, uniq_price_year, 'price', 'price', 'price') #绘制每年的价格变化图表

##########################SuperHost##################
    number=[]
    occupancy_data=pd.concat([data['host_response_rate'],data['host_is_superhost'],data['review_scores_rating']],axis=1)
    for numb in occupancy_data['host_response_rate']:
        number.append(str(numb).split('%')[0].strip())
    occupancy_data['host_response_rate']=number

    # print(occupancy_data.isnull().sum())
    occupancy_data2=occupancy_data.dropna(axis=0,how='any')
    occupancy_data2=occupancy_data2[~occupancy_data2['host_response_rate'].isin(['nan'])]######里面没有空值但是有nan值，删除 #isin()接受一个列表，判断该列中元素是否在列表中，它的反函数就是在前面加上 ~

    superhost_y=occupancy_data2.loc[occupancy_data2['host_is_superhost']=='t',:]
    superhost_f=occupancy_data2.loc[occupancy_data2['host_is_superhost']=='f',:]
    plt.figure()
    plt.scatter(superhost_y['host_response_rate'].astype(int),superhost_y['review_scores_rating'],c='red',label='SuperHost:True',marker='.',alpha=0.5)  #astype()    对数据类型进行转换   
    plt.scatter(superhost_f['host_response_rate'].astype(int),superhost_f['review_scores_rating'],c='blue',label='SuperHost:False',marker='.',alpha=0.5)
    plt.grid(True)
    plt.grid(color='gray')
    plt.grid(linestyle='--')
    plt.xlabel('Host Response Rate')
    plt.ylabel('Avg Rating')
    plt.legend(loc=2)   #图例位置
    plt.show()



def box_plot():
    pass

###################################word_cloud词云制作#####################################
def word_cloud(data):
    word_data=data.copy()
    word_comments=word_data['comments'].tolist()
    chinese_word=[]
    english_word=[]
    for word in word_comments:
        try:
            english_word.append(''.join(byte for byte in word if ord(byte)<256 and byte not in ",!@#$|%^&*()<>?:{}"))##英文文本
            chinese_word.append(''.join(byte.strip() for byte in word.strip() if ord(byte)>256 and byte not in "''!@#$%^&*()_+"))####中文文本
        except:
            pass
    ######英文词云
    w=wordcloud.WordCloud(background_color='white',width=1000,height=700)
    w.generate(str(english_word))
    w.to_file('english_wordcloud.jpg')###保存为jpg图片
    ######中文词云
    ls_chinese=jieba.lcut(str(chinese_word))
    chinese_text=''.join(ls_chinese)
    w2=wordcloud.WordCloud(font_path='simsun.ttc',width=1000,height=700,background_color='white')
    w2.generate(chinese_text)
    w2.to_file('chinese_wordcloud.jpg')

############################决策树预测#########################
def Decision_Tree():
    mix_data=pd.read_csv('decisiontree.csv')####读取数据

    ############################################数据预处理##########################################
    mix_data=mix_data[mix_data['host_is_superhost'].notna()]#host_is_superhost只有一个为空，删除 ##.notna()检测类似数组对象非缺失值
    mix_data['bathrooms']=mix_data['bathrooms'].fillna(mix_data['bathrooms'].mean())#将bathrooms缺失值用均值填充
    mix_data['bedrooms']=mix_data['bedrooms'].fillna(mix_data['bedrooms'].mean())#将bedrooms缺失值用均值填充
    mix_data['beds'] = mix_data['beds'].fillna(mix_data['beds'].mean())#将beds缺失值用均值填充
    security_deposit=[]
    for security in mix_data.loc[mix_data['security_deposit'].notnull(),'security_deposit'].tolist():##将security_deposit指标非空的部分带有美元符号的数值去除美元符号，并设置为float型数值
        security_deposit.append(security.split('$')[1].replace(',','').strip())#将数字中间的逗号去掉（1,456->1456)
    mix_data.loc[mix_data['security_deposit'].notnull(),'security_deposit']=[float(t) for t in security_deposit]###更改数值类型，原先为字符串，改为float
    mix_data['security_deposit']=mix_data['security_deposit'].fillna(mix_data['security_deposit'].mean())#用处理好为数值型的数字的平均值填充空值
    cleaning_fee=[]#############cleaning_fee指标的处理方法同上
    for cleaning in mix_data.loc[mix_data['cleaning_fee'].notnull(),'cleaning_fee'].tolist():
        cleaning_fee.append(cleaning.split('$')[1].replace(',','').strip())
    mix_data.loc[mix_data['cleaning_fee'].notnull(),'cleaning_fee']=[float(t) for t in cleaning_fee]
    mix_data['cleaning_fee']=mix_data['cleaning_fee'].fillna(mix_data['cleaning_fee'].mean())
    # print(cleaning_fee)
    # mix_data['security_deposit'] = mix_data['security_deposit'].fillna(mix_data['security_deposit'].astype(float).mean())
    for nadata in ['review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_value','review_scores_location','reviews_per_month']:
        mix_data[nadata]=mix_data[nadata].fillna(mix_data[nadata].mean())#############将上述指标的空值分别用均值填充
        # print('属性{}已完成填充'.format(nadata))
    mix_data=mix_data[~mix_data['host_response_rate'].isnull()]#####将host_response_time属性为空值的删除，只删除4000左右，对结果影响不大
    mix_data.index=range(mix_data.shape[0])
    ########################################至此空值已全部处理完#############################

    # print(mix_data.isnull().sum())
    # print(mix_data.shape)
    mix_data.drop_duplicates(inplace=True)###去除重复值

    list_=list(mix_data['host_is_superhost'])
    for i,j in enumerate(list_):####################将host_is_superhost指标的t改为1，f改为0
        if j=='t':
            list_[i]=1
        else:
            list_[i]=0
    mix_data['host_is_superhost']=list_

    ##############将neighbourhood_cleansed特征哑变量处理##############################

        
    mix_data['neighbourhood_cleansed']=[neibour.split('/')[0].strip() for neibour in mix_data['neighbourhood_cleansed']]
    One_hot_encoder = OneHotEncoder().fit(mix_data['neighbourhood_cleansed'].values.reshape(-1,1)) #OneHotEncoder 函数非常实用，它可以实现将分类特征的每个元素转化为一个可以用来计算的值
    One_data=OneHotEncoder().fit_transform(mix_data['neighbourhood_cleansed'].values.reshape(-1,1)).toarray()
    neibourhood_cleansed=pd.DataFrame(One_data)
    neibourhood_cleansed.columns=One_hot_encoder.get_feature_names() #get_feature_names():返回一个含有特征名称的列表，通过索引排序，如果含有one-hot表示的特征，则显示相应的特征名
    neibourhood_cleansed.index=range(neibourhood_cleansed.shape[0])
    mix_data=pd.concat([neibourhood_cleansed,mix_data.drop(['neighbourhood_cleansed'],axis=1)],axis=1)
    # print(neibourhood_cleansed)

    dataframe=[]
    features=['property_type','room_type','bed_type','cancellation_policy']
    for feature in features:
        one_hot_label=OneHotEncoder().fit(mix_data[feature].values.reshape(-1,1))
        one_hot_data=OneHotEncoder().fit_transform(mix_data[feature].values.reshape(-1,1)).toarray() #将list直接转为Object[] 数组
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
    ###########################################至此全部特征转换为数值型############################################

    y=pd.DataFrame(mix_data.loc[:,'price'])#############标签列
    x=mix_data.drop(['price'],axis=1)###############特征列
    # print(x.info)
    x=x.astype('float')
    # y=y.astype('int')
    # print(x.dtypes)
    ###特征的数据处理----标准化Z-scores####
    # x=(x-x.mean())/np.std(x)*100

    x.to_csv('decisiontree_dealdata.csv')###################所有处理好的数据 保存进csv文件
   #     # print(mix_data['host_is_superhost'])
    model=DecisionTreeRegressor(random_state=30,criterion='mse',max_depth=10) #random_state指定随机数生成时所用算法开始的整数值
    data_split=int(x.shape[0]*0.7)
    # print(data_split)
    x_train,x_test,y_train,y_test=x.iloc[:data_split,:],x.iloc[data_split:,:],y.iloc[:data_split,:],y.iloc[data_split:,:]##划分测试集和训练集，因为数据集本身没有顺序，所以取前70%作为训练集，后30%作为测试集
    x_test.index,y_test.index=range(x_test.shape[0]),range(y_test.shape[0])###重置索引 一定要重置！！！！！
    model.fit(x_train,y_train)#模型的fit
    predict=model.predict(x_test)#对测试集的预测
    convent=pd.concat([pd.DataFrame(predict),y_test],axis=1)
    convent.columns=['prediction value','real value']
    #降维前的绝对平均误差
    print('Mean Absolute Deviation without dimensionality reduction：', np.abs(convent['prediction value']-convent['real value']).mean())#降维前的绝对平方误差
    # print(convent)
    convent.to_csv('未降维时的预测比较.csv')######################################特征矩阵的特征数太多，预测效果不是很好，降维处理
    plt.figure()
    plt.plot(range(y_test.shape[0]),predict,c='red',label='predict price')
    plt.plot(range(y_test.shape[0]),y_test,c='lightblue',label='real price')
    plt.title('compare with prediction without dimensionality reduction')
    plt.legend(loc=2)
    plt.show()

    ###降维
    x_score=[]
    for i in range(1,20):###找到最佳降维后的特征数
        pca_x=PCA(n_components=i,svd_solver='full').fit(x) #即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的。有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。randomized一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。 full则是传统意义上的SVD，使用了scipy库对应的实现。arpack和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，而arpack直接使用了scipy库的sparse SVD实现。默认是auto，即PCA类会自己去在前面讲到的三种算法里面去权衡，选择一个合适的SVD算法来降维。一般来说，使用默认值就够了。
        # print(pca_x.explained_variance_ratio_)
        x_score.append(pca_x.explained_variance_ratio_.sum())#explained_variance_ratio_它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。   #explained_variance_ratio_计算了每个特征方差贡献率，所有总和为1，explained_variance_为方差值，通过合理使用这两个参数可以画出方差贡献率图或者方差值图，便于观察PCA降维最佳值
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.figure()
    plt.plot(range(len(x_score)),x_score,c='red',label='trend of explained variance ratio with changing number of attribute',linewidth=4) #特征方差贡献率随降维后的特征数的走势
    plt.title('trend graph of explained variance ratio with changing number of attribute')
    plt.legend(loc=4)
    plt.show()
    ####观察了图像后，发现在8类别时方差不在变化，于是我们选择聚为8类###############
    pca_best=PCA(n_components=8,svd_solver='full').fit_transform(x)#降维降为8个特征属性
    x_best=pd.DataFrame(pca_best)
    x_best_train,x_best_test,y_best_train,y_best_test=x_best.iloc[:data_split,:],x_best.iloc[data_split:,:],y.iloc[:data_split,:],y.iloc[data_split:,:]
    x_best_test.index,y_best_test.index=range(x_test.shape[0]),range(y_test.shape[0])
    model.fit(x_best_train,y_best_train)
    best_predict=model.predict(x_best_test)
    best_convent=pd.concat([pd.DataFrame(best_predict),pd.DataFrame(y_best_test)],axis=1)
    best_convent.columns=['prediction value','real value']
    #降维后的绝对平均误差
    print('Mean Absolute Deviation with dimensionality reduction：', np.abs(best_convent['prediction value'] - best_convent['real value']).mean())
    best_convent.to_csv('降维后的预测比较.csv')
    plt.figure()
    plt.plot(range(best_predict.shape[0]),best_predict,c='red',label='prediction value')
    plt.plot(range(y_best_test.shape[0]),y_best_test,c='lightblue',label='real value')
    plt.title('compare with prediction with dimensionality reduction')
    plt.legend(loc=2)
    plt.show()


def K_means(data):###聚类
    k_data=data[['accommodates','price','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','beds','bedrooms']]
    # print(k_data)
    #数据处理
    k_data=k_data.fillna(k_data.mean())##用均值填充空值
    #区名中文变拼音
    pinname = Pinyin()
    k_data['area'] = [pinname.get_pinyin(area.split('/')[0].strip(), '')[:-2] for area in data['neighbourhood_cleansed']]
    x=k_data.drop(['price'],axis=1)#特征矩阵，将地区一列去掉
    # print(x.isnull().sum())
    y=k_data.loc[:,'price']
    silh_score=[]#轮廓系数，用轮廓系数来判断最佳的聚类总数，轮廓系数越大，越优，即最优的簇数（k值）。轮廓系数的值是介于 [-1,1] ，越趋近于1代表内聚度和分离度都相对较优

    color = ['red', 'blue', 'green', 'black', 'pink', 'purple', 'yellow']
    for i in range(2,7):#找到最佳聚类的k值，画学习曲线的方式，聚类总数至少为2类
        model=KMeans(n_clusters=i).fit(k_data.drop(['area'],axis=1))#模型的fit
        y_predict=model.predict(k_data.drop(['area'],axis=1))#训练模型后的聚类结果
        silh_score.append(silhouette_score(x.drop(['area'],axis=1),y_predict))###将不同k值下的轮廓系数
        print('done number {}'.format(i))##############检验程序进行到哪一步了
        plt.figure(figsize=(9,6))
        for j in range(i):
            k_data['label']=model.labels_
            plt.scatter(k_data.loc[k_data['label']==j]['area'],k_data.loc[k_data['label']==j]['price'],marker='+',c=color[j])
        plt.title('the clustering effect when k value is {}'.format(i))
        plt.xlabel('area')
        plt.ylabel('price')
        plt.show()


    best_score=silh_score.index(max(silh_score))+2#以最大的轮廓系数下的k值聚类，并可视化
    print('Best number of clustering：',best_score)
    plt.figure()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.plot(range(2,len(silh_score)+2),silh_score,c='black')
    plt.title('Silhouette Coefficient trend with changing number of clustering')
    plt.show()
################以最佳的簇数即k值来画图
    model2=KMeans(n_clusters=best_score)
    model2.fit(k_data.drop(['area'],axis=1))
    k_data['label']=model.labels_#把预测标签加到数据集中
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

'''决策树：1：数据预处理。
            首先数据的去重。再针对数值型数值的特征：用均值填充。针对字符型的数值的特征，比如带有美元符号或者百分号，将符号去掉，并将数值中的逗号去掉，并astype的方法设置为float
            型数值，再用处理好的数值用均值填充空值。对字符串型的特征，比如地区、host_is_superhost等，哑变量处理。
           2：划分测试集和训练集，因为数据集本身无序性，所以取前70%为训练集，余下的为测试集。
           3：模型的建立  4：模型对测试集的预测并与原数据比较。
           
    k_means聚类：首先也是数据预处理，用均值填充空值。为了判断最佳的聚类的k值，用轮廓系数衡量，轮廓系数越大，聚类效果越好。在这里利用
                    画学习曲线的方法，找到最大轮廓系数下的k值。再以最佳k值聚类，并可视化聚类效果'''