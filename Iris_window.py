from tkinter import *
import tkinter.messagebox as m
###importing from sklearn database
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=6)
####window design
w=Tk()
w.title("CLASSIFICATION OF IRIS FLOWER USING MACHINE LEARNING")
w.configure(bg='lightblue')
sl=StringVar()
sw=StringVar()
pl=StringVar()
pw=StringVar()

##knn
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)
acc_knn=accuracy_score(y_test,y_pred_knn)
##nb
nb=GaussianNB()
nb.fit(x_train,y_train)
y_pred_nb=nb.predict(x_test)
acc_nb=accuracy_score(y_test,y_pred_nb)
###lg
lg=LogisticRegression(solver='liblinear',multi_class='auto')
lg.fit(x_train,y_train)
y_pred_lg=lg.predict(x_test)
acc_lg=accuracy_score(y_test,y_pred_lg)
##dt
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(x_test)
acc_dt=accuracy_score(y_test,y_pred_dt)




###drawing bar chart
def compare():
    import matplotlib.pyplot as plt;plt.rcdefaults()
    try:
        objects=('KNN','LogisticRegression','Naive Bayes','Decision Tree')
        y_pos=np.arange(len(objects))
        accuracy=[round(acc_knn*100,2),round(acc_lg*100,2),round(acc_nb*100,2),round(acc_dt*100,2)]
        plt.bar(y_pos,accuracy,align='center',alpha=0.5,color=['green','red','yellow','blue'])
        plt.xticks(y_pos,objects)
        plt.ylabel('ACCURACY')
        plt.title('comparision barchart of accuracy of different models')
        plt.show()
    except:
        m.showwarning(title="WARNING",message="please click on all the models first")

def reset():
    sl.set('')
    sw.set("")
    pl.set("")
    pw.set("")
def pr_knn():
    m.showinfo(title="KNN",message="accuracy of knn model is {}".format(round(acc_knn*100,2)))    
    
def pr_nb():
    m.showinfo(title="NAIVE BAYES",message="accuracy of Naive Bayes model is {}".format(round(acc_nb*100,2)))
    
def pr_dt():
    m.showinfo(title="DECISION TREE",message="accuracy of Decision Tree model is {}".format(round(acc_dt*100,2)))
    
def pr_lg():
    m.showinfo(title="LOGISTIC REGRESSION",message="accuracy of Logistic Regression model is {}".format(round(acc_lg*100,2)))
    
           
    
def submit():
    try:
        check=max([float(acc_knn),float(acc_nb),float(acc_lg),float(acc_dt)])
        if(acc_knn==check):
            final=knn.predict([[float(sl.get()),float(sw.get()),float(pl.get()),float(pw.get())]])
        elif(acc_lg==check):
            final=lg.predict([[float(sl.get()),float(sw.get()),float(pl.get()),float(pw.get())]])
        elif(acc_nb==check):
            final=nb.predict([[float(sl.get()),float(sw.get()),float(pl.get()),float(pw.get())]])
        else:
            final=dt.predict([[float(sl.get()),float(sw.get()),float(pl.get()),float(pw.get())]])

        if final==[0]:
            m.showinfo(title='prediction',message='the flower belongs to class iris setosa ')
        elif final==[1]:
            m.showinfo(title='prediction',message='the flower belongs to class iris versicolor')
        else:
            m.showinfo(title='prediction',message='the flower belongs to class iris virginia')
    except ValueError:
        m.showwarning(title="ERROR",message="please enter valid parameters")
    except:
        m.showwarning(title="ERROR",message="please click on all the models first")
    
b_knn=Button(w,relief='groove',text='KNN',font=('calibri',20,'bold'),width=16,command=pr_knn,bg='blue')
b_nb=Button(w,relief='groove',text='Naive Bayes',font=('calibri',20,'bold'),width=16,command=pr_nb,bg='blue')
b_dt=Button(w,relief='groove',text='Decision Tree',font=('calibri',20,'bold'),width=16,command=pr_dt,bg='blue')
b_lg=Button(w,relief='groove',text='Logistic Regression',font=('calibri',20,'bold'),width=16,command=pr_lg,bg='blue')
l_show=Label(w,text="Enter Details For A New Flower",font=('calibri',20,'bold'),bg='lightblue',fg='green')
l_sl=Label(w,text='sepal length',font=('calibri',20,'bold'),bg='lightblue')
l_sw=Label(w,text='sepal width',font=('calibri',20,'bold'),bg='lightblue')
l_pl=Label(w,text='petal length',font=('calibri',20,'bold'),bg='lightblue')
l_pw=Label(w,text='petal width',font=('calibri',20,'bold'),bg='lightblue')
b_submit=Button(w,relief='groove',text='SUBMIT',font=('arial',20,'bold'),bg='blue',command=submit)
b_reset=Button(w,relief='groove',text='RESET',font=('arial',20,'bold'),command=reset,bg='blue')
b_comp=Button(w,relief='groove',text='COMPARE',font=('calibri',20,'bold'),width=16,command=compare,bg='blue')
e_sl=Entry(w,textvariable=sl,relief='groove')
e_sw=Entry(w,textvariable=sw,relief='groove')
e_pl=Entry(w,textvariable=pl,relief='groove')
e_pw=Entry(w,textvariable=pw,relief='groove')

b_knn.grid(row=1,column=1,pady=20,padx=10)
b_lg.grid(row=2,column=1,pady=20,padx=10)
b_nb.grid(row=3,column=1,pady=20,padx=10)
b_dt.grid(row=4,column=1,pady=20,padx=10)
b_comp.grid(row=5,column=1,padx=10)
l_show.grid(row=1,column=2,columnspan=2)
l_sl.grid(row=2,column=2,pady=20)
l_sw.grid(row=3,column=2,pady=20)
l_pl.grid(row=4,column=2,pady=20)
l_pw.grid(row=5,column=2,pady=20)
b_submit.grid(row=6,column=2)
b_reset.grid(row=6,column=3)
e_sl.grid(row=2,column=3)
e_sw.grid(row=3,column=3)
e_pl.grid(row=4,column=3)
e_pw.grid(row=5,column=3)
w.mainloop()

