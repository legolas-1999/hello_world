from tkinter import *
import tkinter.messagebox as m
###importing from sklearn database
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
######new project pandas code
import pandas as pd
#col_names=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima=pd.read_csv("D:/pima.csv")
feature=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age']
x=pima[feature]
y=pima.output
####cleaning of data
feature.pop(0)
for i in feature:
    med_i=pima[i].median()
    pima[i]=pima[i].replace(to_replace=0,value=med_i)

feature.insert(0,'pregnant')
x=pima[feature]
y=pima.output
########
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=6)
####window design
w=Tk()
w.title("PREDICTION OF DIABETES")
w.configure(bg='lightblue')
preg=StringVar()
glu=StringVar()
bp=StringVar()
skin=StringVar()
ins=StringVar()
ped=StringVar()
bmi=StringVar()
age=StringVar()

##knn
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)
acc_knn=accuracy_score(y_test,y_pred_knn)
confusion=metrics.confusion_matrix(y_test,y_pred_knn)
print(type(confusion))
print(confusion)
print(confusion[1][1],confusion[1,1])
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
        plt.ylabel('ACCURACy')
        plt.title('comparision barchart of accuracy of different models')
        plt.show()
    except:
        m.showwarning(title="WARNING",message="please click on all the models first")

def reset():
    preg.set('')
    glu.set("")
    bp.set("")
    skin.set("")
    ped.set('')
    ins.set("")
    bmi.set("")
    age.set("")
def pr_knn():
    m.showinfo(title="KNN",message="accuracy of knn model is {}".format(round(acc_knn*100,2)))    
    
def pr_nb():
    m.showinfo(title="NAIVE BAyES",message="accuracy of Naive Bayes model is {}".format(round(acc_nb*100,2)))
    
def pr_dt():
    m.showinfo(title="DECISION TREE",message="accuracy of Decision Tree model is {}".format(round(acc_dt*100,2)))
    
def pr_lg():
    m.showinfo(title="LOGISTIC REGRESSION",message="accuracy of Logistic Regression model is {}".format(round(acc_lg*100,2)))
    
           
    
def submit():
    try:
        check=max([float(acc_knn),float(acc_nb),float(acc_lg),float(acc_dt)])
        if(acc_knn==check):
            final=knn.predict([[float(preg.get()),float(glu.get()),float(bp.get()),float(skin.get()),float(ins.get()),float(bmi.get()),float(ped.get()),float(age.get())]])
        elif(acc_lg==check):
            final=lg.predict([[float(preg.get()),float(glu.get()),float(bp.get()),float(skin.get()),float(ins.get()),float(bmi.get()),float(ped.get()),float(age.get())]])
        elif(acc_nb==check):
            final=nb.predict([[float(preg.get()),float(glu.get()),float(bp.get()),float(skin.get()),float(ins.get()),float(bmi.get()),float(ped.get()),float(age.get())]])
        else:
            final=dt.predict([[float(preg.get()),float(glu.get()),float(bp.get()),float(skin.get()),float(ins.get()),float(bmi.get()),float(ped.get()),float(age.get())]])

        if final==0:
            m.showinfo(title='prediction',message='the person does not have diabetes ')
        else :
            m.showinfo(title='prediction',message='the person has diabetes')
        
    except ValueError:
        m.showwarning(title="ERROR",message="please enter valid parameters")
    except:
        m.showwarning(title="ERROR",message="please click on all the models first")
    
b_knn=Button(w,relief='groove',text='KNN',font=('calibri',20,'bold'),width=16,command=pr_knn,bg='blue')
b_nb=Button(w,relief='groove',text='Naive Bayes',font=('calibri',20,'bold'),width=16,command=pr_nb,bg='blue')
b_dt=Button(w,relief='groove',text='Decision Tree',font=('calibri',20,'bold'),width=16,command=pr_dt,bg='blue')
b_lg=Button(w,relief='groove',text='Logistic Regression',font=('calibri',20,'bold'),width=16,command=pr_lg,bg='blue')
l_show=Label(w,text="Enter Details About The Patient",font=('calibri',20,'bold'),bg='lightblue',fg='green')
l_preg=Label(w,text='pregnant',font=('calibri',20,'bold'),bg='lightblue')
l_glu=Label(w,text='glucose',font=('calibri',20,'bold'),bg='lightblue')
l_bp=Label(w,text='bp',font=('calibri',20,'bold'),bg='lightblue')
l_skin=Label(w,text='skin',font=('calibri',20,'bold'),bg='lightblue')
l_ins=Label(w,text='insulin',font=('calibri',20,'bold'),bg='lightblue')
l_bmi=Label(w,text='bmi',font=('calibri',20,'bold'),bg='lightblue')
l_ped=Label(w,text='pedigree',font=('calibri',20,'bold'),bg='lightblue')
l_age=Label(w,text='age',font=('calibri',20,'bold'),bg='lightblue')
b_submit=Button(w,relief='groove',text='SUBMIT',font=('arial',20,'bold'),bg='blue',command=submit)
b_reset=Button(w,relief='groove',text='RESET',font=('arial',20,'bold'),command=reset,bg='blue')
b_comp=Button(w,relief='groove',text='COMPARE',font=('calibri',20,'bold'),width=16,command=compare,bg='blue')
e_bmi=Entry(w,textvariable=bmi,relief='groove')
e_age=Entry(w,textvariable=age,relief='groove')
e_preg=Entry(w,textvariable=preg,relief='groove')
e_ins=Entry(w,textvariable=ins,relief='groove')
e_ped=Entry(w,textvariable=ped,relief='groove')
e_skin=Entry(w,textvariable=skin,relief='groove')
e_glu=Entry(w,textvariable=glu,relief='groove')
e_bp=Entry(w,textvariable=bp,relief='groove')

b_knn.grid(row=1,column=1,pady=20,padx=10)
b_lg.grid(row=2,column=1,pady=20,padx=10)
b_nb.grid(row=3,column=1,pady=20,padx=10)
b_dt.grid(row=4,column=1,pady=20,padx=10)
b_comp.grid(row=5,column=1,padx=10)
l_show.grid(row=1,column=2,columnspan=4)
l_preg.grid(row=2,column=2,pady=20)
l_glu.grid(row=3,column=2,pady=20)
l_bp.grid(row=4,column=2,pady=20)
l_skin.grid(row=5,column=2,pady=20)
b_submit.grid(row=6,column=2,columnspan=2)
b_reset.grid(row=6,column=4,columnspan=2)
e_preg.grid(row=2,column=3)
e_glu.grid(row=3,column=3)
e_bp.grid(row=4,column=3)
e_skin.grid(row=5,column=3)
l_ins.grid(row=2,column=4,pady=20)
e_ins.grid(row=2,column=5)
l_bmi.grid(row=3,column=4)
e_bmi.grid(row=3,column=5)
l_ped.grid(row=4,column=4)
e_ped.grid(row=4,column=5)
l_age.grid(row=5,column=4)
e_age.grid(row=5,column=5)
w.mainloop()

