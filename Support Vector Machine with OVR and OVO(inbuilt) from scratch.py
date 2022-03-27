class SVClassifier():
    "Support Vector Machine Classifier which uses the fit function of sklearn and predict on the basis of Decision function."
    def __init__(self,C=1.0, multiclass=None, kernel ='rbf'):
        self.C = C
        self.multi_class = multiclass
        self.kernel = kernel
        
    def fit(self, X,y, columns):
        "fit() of sklearn is used to predict the outcomes."
        self.X = X
        self.y = y
       # print(self.y)
        self.df = pd.DataFrame(self.X, columns =list(columns)[:-1])
        self.df['labels'] = self.y
        self.list_of_models=[]
        self.pairs_of_binary=[]
        if self.multi_class == 'ovo':
            self.n_classes = (len(np.unique(self.y))*(len(np.unique(self.y))-1))//2 
            #print(self.n_classes)
            for class1 in range(self.n_classes-1):
                for class2 in range(class1,self.n_classes):
                    if class1!=class2:
                        self.df1 = self.df.loc[(self.df['labels']==class1) | (self.df['labels']==class2)]
                        self.X = self.df1.drop(columns ='labels')
                        self.y = self.df1['labels']
                        #print(self.y)
                        svm = SVC(C= self.C, kernel =self.kernel)
                        svm.fit(self.X, self.y)
                        self.pairs_of_binary.append((class1, class2))
                        self.list_of_models.append(svm)
        elif self.multi_class =='ovr':
            self.n_classes = len(np.unique(self.y))
            self.binary_class = (1,-1)
            for bin_class in range(self.n_classes):
                self.X = self.df.drop(columns ='labels')
                self.y = np.where(self.df['labels']==bin_class,1, -1)
                svc = SVC(C=self.C,kernel =self.kernel)
                svc.fit(self.X, self.y)
                self.list_of_models.append(svc)
        return f'no of models created:{self.n_classes}'
        
    def majority_votes_identifier(self,models_pred):
        "To find out the which class has the majority votes for the labels. Only for OVO(One v/s One)"
        y_pred = []
        for each_instance in models_pred:
            unique , counts = np.unique(each_instance,return_counts=True)
            unique, counts = list(unique), list(counts)
            index = counts.index(max(counts))
            y_pred.append(int(unique[index]))
        return np.array(y_pred)
    
    
    def predict(self, x_test):
        "To find the predictions for the test data."
        self.x_test=x_test
        models_pred = np.zeros((len(x_test),self.n_classes))
        if self.multi_class=='ovo':
            for i in range(self.n_classes):
                decision_func = self.list_of_models[i].decision_function(self.x_test)
                max_label = max(self.pairs_of_binary[i][0], self.pairs_of_binary[i][1])
                min_label = min(self.pairs_of_binary[i][0], self.pairs_of_binary[i][1])
                y_pred = np.where(decision_func>0, max_label, min_label)
                models_pred[:,i]= y_pred
            self.y_pred=self.majority_votes_identifier(models_pred)
        elif self.multi_class=='ovr':
            for i in range(self.n_classes):
                decision_func = self.list_of_models[i].decision_function(self.x_test)
                max_label = max(self.binary_class)
                min_label = min(self.binary_class)
                y_pred = np.where(decision_func>0,max_label, min_label)
                models_pred[:,i] = y_pred
            self.y_pred = np.argmax(models_pred, axis=1)
        return self.y_pred
    
    def accuracy_score(self, y_test, y_pred):
        "To find the accuracy of the model."
        correctly_predicted=0
        for i, j in zip(y_test, y_pred):
            if i==j:
                correctly_predicted+=1
        accuracy = (correctly_predicted)/len(y_test)
        return accuracy
                    
    def class_accuracy(self, y_test, y_pred):
        "To compute accuracy on the basis of Class or labels."
        if self.multi_class =='ovo' or self.multi_class=='ovr':
            range_of_list = len(np.unique(y_test))
            class_wise_accuracy={}
            y_test1 = pd.Series(y_test)
            for i in range(range_of_list):
                y_test1_index = y_test1[y_test1==i].index
                y_pred1 = y_pred[y_test1_index]
                class_wise_accuracy[f'Class {i}'] = (list(y_pred1).count(i)/list(y_test).count(i))
        return class_wise_accuracy
