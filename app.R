library(caret)
#install.packages("caret")
library(ggplot2)
#install.packages("ggplot2")
library(dplyr)
#install.packages("dplyr")
library(shiny)
#install.packages("shiny")
library(pROC)
#install.packages("pROC")
library(xgboost)
#install.packages("xgboost")
library(plotly)
#install.packages("plotly")
library(plyr)
#install.packages("plyr")
library(e1071)
#install.packages("e1071")
library(shinyWidgets)
#install.packages("shinyWidgets")
library(rsconnect)
#install.packages("rsconnect")
library(shinyjs)
#install.packages("shinyjs")
library(cluster)
#install.packages("cluster")
library(flexclust)
#load data

data <- read.csv('heart.csv', header = TRUE)
#testMbr <- read.csv('testMbr.csv', header = TRUE)
#write.csv(train2, file = 'train2.csv')

str(data)

data$sex <- as.factor(data$sex)
data$cp <- as.factor(data$cp) #Chest pain
data$fbs <- as.factor(data$fbs) #fasting? Y/N
data$restecg <- as.factor(data$restecg) #resting ecg
data$exng <- as.factor(data$exng) #Exercise induced angina
data$thall <- as.factor(data$thall) #
data$output <- as.factor(data$output) # output
data$slp <- as.factor(data$slp)
data$caa <- as.factor(data$caa)


str(data)

#one-hot encoding
dummies <- dummyVars( ~ . , data = data)
ex <- data.frame(predict(dummies, newdata = data))
names(ex) <- gsub("\\.","",names(ex))
d <- cbind(data$output,ex)
names(d)[1] <- "y"

#Clean up
rm(dummies, ex)
str(d)

#Remove highly correlated columns
descrCor <- cor(d[,2:ncol(d)])  # find the correlation of all inputs except y (predictor)

highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.80) # Identify higly correlated columns
filterDescr <- d[,2:ncol(d)][,-highlyCorDescr] # Create a data frame and exclude correlations
descrCor2 <- cor(filterDescr)

summary(descrCor2[upper.tri(descrCor2)])  # check if the correlation is within +/- 0.80

d <- cbind(d$y, filterDescr)
names(d)[1] <- "y"

#Clean up
rm(descrCor, descrCor2, filterDescr)


#Identify Linear Dependencies
y <- d$y
d <- cbind(rep(1,nrow(d)),d[2:ncol(d)])
names(d)[1] <- "ones"
(comboInfo <- findLinearCombos(d)) # Idnetify linear combinations

d <- d[,-comboInfo$remove] # Remove columns with high collinearity
d <- d[,c(2:ncol(d))]
d <- cbind(y,d)

rm(y,comboInfo)

#Remove columns with little variation
nzv <- nearZeroVar(d,saveMetrics = TRUE)
str(nzv)
(d <- d[, c(TRUE,!nzv$nzv[2:ncol(d)])])

rm(nzv)

#Min-Max NOrmalization
#preProc_d <- preProcess(d[,c(2:ncol(d))], method = c("range"))
#d <- predict(preProc_d,d)

#rm(preProc_d)

#####################################################################################################################
set.seed(1234)
intrain <- createDataPartition(y=d$y, p=.5, list=F)

train <- d[intrain,]
test <- d[-intrain,]
#Create K-Means Clusters
k_df <- data.frame() #accumulator for cluster results
#k_df
#allow up to 50 iterations to obtain convergence, and do 20 random starts
# train set
?kmeans
train[,2:ncol(train)]
for(k in 1:15){
  kmeans_tr <- kmeans(x=train[,2:ncol(train)], centers=k, nstart=20, iter.max=50)
  # test set
  kmeans_te <- kmeans(x=test[,2:ncol(test)], centers=k, nstart=20, iter.max=50)
  
  #Combine cluster number and cost together, write to df
  k_df <- rbind(k_df, cbind(k, kmeans_tr$tot.withinss
                            , kmeans_te$tot.withinss))
}
# the k_df data.frame contains the # of clusters k and the MSE for each cluster
names(k_df) <- c("cluster", "tr_k", "te_k")
k_df

# create an elbow plot
par(mfrow=c(1,1))
k_df[,2:3] <- k_df[,2:3]/1000
plot(x=k_df$cluster, y=k_df$tr_k, main="k-Means Elbow Plot"
     , col="blue", pch=19, type="b", cex.lab=1.2
     , xlab="Number of Clusters", ylab="MSE (in 1000s)")
points(x=k_df$cluster, y=k_df$te_k, col="green")
#k_df

#Reclassify Using 4,5,6 Clusters & test
#library(flexclust)
test0 <- test
test1 <- test
test2 <- test
kmeans_tr3 <- kcca(x=train[,2:ncol(train)], k=5, family=kccaFamily("kmeans"))
test0$clust3 <- predict(kmeans_tr3,test0[,2:ncol(test0)])

kmeans_tr <- kcca(x=train[,2:ncol(train)], k=5, family=kccaFamily("kmeans"))
test$clust5 <- predict(kmeans_tr,test[,2:ncol(test)])

kmeans_tr4 <- kcca(x=train[,2:ncol(train)], k=4, family=kccaFamily("kmeans"))
test1$clust4 <- predict(kmeans_tr4,test1[,2:ncol(test1)])

kmeans_tr6 <- kcca(x=train[,2:ncol(train)], k=6, family=kccaFamily("kmeans"))
test2$clust6 <- predict(kmeans_tr6,test2[,2:ncol(test2)])
# average silhouette values by cluster on test data set
km3_train <- data.frame(rep("train k=3",nrow(kmeans_tr3@"centers"))
                        ,cbind(c(1:nrow(kmeans_tr3@"centers")), kmeans_tr3@"centers"))
km5_train <- data.frame(rep("train k=5",nrow(kmeans_tr@"centers"))
                        ,cbind(c(1:nrow(kmeans_tr@"centers")), kmeans_tr@"centers"))
km4_train <- data.frame(rep("train k=4",nrow(kmeans_tr4@"centers"))
                        ,cbind(c(1:nrow(kmeans_tr4@"centers")), kmeans_tr4@"centers"))
km6_train <- data.frame(rep("train k=6",nrow(kmeans_tr6@"centers"))
                        ,cbind(c(1:nrow(kmeans_tr6@"centers")), kmeans_tr6@"centers"))
km3_test <- data.frame(rep("test k=3",nrow(kmeans_tr3@"centers")),
                       aggregate(test0[,2:ncol(test0)], by=list(test0[,"clust3"]), FUN=mean))
km5_test <- data.frame(rep("test k=5",nrow(kmeans_tr@"centers")),
                       aggregate(test[,2:ncol(test)], by=list(test[,"clust5"]), FUN=mean))
km4_test <- data.frame(rep("test k=4",nrow(kmeans_tr4@"centers")),
                       aggregate(test1[,2:ncol(test1)], by=list(test1[,"clust4"]), FUN=mean))
km6_test <- data.frame(rep("test k=6",nrow(kmeans_tr6@"centers")),
                       aggregate(test2[,2:ncol(test2)], by=list(test2[,"clust6"]), FUN=mean))
par(mfrow = c(2,2))

km3 <- kmeans(train[,2:ncol(train)], 3)
dist3 <- dist(train[,2:ncol(train)], method="euclidean")
sil3 <- silhouette(km3$cluster, dist3)
plot(sil3, col=c("black","red","green"), main="Silhouette plot Train (k=3) K-means")
km3t <- kmeans(test0[,2:ncol(test0)], 3)
dist3t <- dist(test0[,2:ncol(test0)], method="euclidean")
sil3t <- silhouette(km3t$cluster, dist3t)
plot(sil3t, col=c("black","red","green"), main="Silhouette plot Test (k=3) K-means")

km4 <- kmeans(train[,2:ncol(train)], 4)
dist4 <- dist(train[,2:ncol(train)], method="euclidean")
sil4 <- silhouette(km4$cluster, dist4)
plot(sil4, col=c("black","red","green","blue"), main="Silhouette plot Train (k=4) K-means")
km4t <- kmeans(test[,2:ncol(test)], 4)
dist4t <- dist(test[,2:ncol(test)], method="euclidean")
sil4t <- silhouette(km4t$cluster, dist4t)
plot(sil4t, col=c("black","red","green","blue"), main="Silhouette plot Test (k=4) K-means")

km5 <- kmeans(train[,2:ncol(train)], 5)
dist5 <- dist(train[,2:ncol(train)], method="euclidean")
sil5 <- silhouette(km5$cluster, dist5)
plot(sil5, col=c("black","red","green","blue","grey"), main="Silhouette plot Train (k=5) K-means")
km5t <- kmeans(test[,2:ncol(test)], 5)
dist5t <- dist(test[,2:ncol(test)], method="euclidean")
sil5t <- silhouette(km5t$cluster, dist5t)
plot(sil5t, col=c("black","red","green","blue","grey"), main="Silhouette plot Test (k=5) K-means")

km6 <- kmeans(train[,2:ncol(train)], 6)
dist6 <- dist(train[,2:ncol(train)], method="euclidean")
sil6 <- silhouette(km6$cluster, dist6)
plot(sil6, col=c("black","red","green","blue","grey","purple"), main="Silhouette plot Train (k=6) K-means")
km6t <- kmeans(test[,2:ncol(test)], 6)
dist6t <- dist(test[,2:ncol(test)], method="euclidean")
sil6t <- silhouette(km6t$cluster, dist6t)
plot(sil6t, col=c("black","red","green","blue","grey","purple"), main="Silhouette plot Test (k=6) K-means")
par(mfrow=c(1,1))

##Result points to 
##################################################################################
names(km3_train)[1:2] <- c("Dataset","Cluster")
names(km3_train)
names(km5_train)[1:2] <- c("Dataset","Cluster")
names(km4_train)[1:2] <- c("Dataset","Cluster")
names(km6_train)[1:2] <- c("Dataset","Cluster")
names(km3_test)[1:2] <- c("Dataset","Cluster")
names(km3_test)
names(km5_test)[1:2] <- c("Dataset","Cluster")
names(km4_test)[1:2] <- c("Dataset","Cluster")
names(km6_test)[1:2] <- c("Dataset","Cluster")
#names(km5_train)[ncol(km5_train)]
#names(km5_test)[ncol(km5_test)-1]
results <- rbind(km5_train,km5_test[,1:ncol(km5_test)-1],
                 km3_train,km3_test[,1:ncol(km3_test)-1],
                 km4_train,km4_test[,1:ncol(km4_test)-1],
                 km6_train,km6_test[,1:ncol(km6_test)-1])
str(results)

#visualize clusters
#source("multiplot.R")
#library(ggplot2)
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


#visualize the clusters
names(results)
p1=ggplot(results, aes(x=Dataset, y=age, fill=Cluster)) + geom_bar(stat="identity")
p2=ggplot(results, aes(x=Dataset, y=sex1, fill=Cluster)) + geom_bar(stat="identity")
p3=ggplot(results, aes(x=Dataset, y=cp0, fill=Cluster)) + geom_bar(stat="identity")
p4=ggplot(results, aes(x=Dataset, y=cp1, fill=Cluster)) + geom_bar(stat="identity")
p5=ggplot(results, aes(x=Dataset, y=cp2, fill=Cluster)) + geom_bar(stat="identity")
p6=ggplot(results, aes(x=Dataset, y=trtbps, fill=Cluster)) + geom_bar(stat="identity")
p7=ggplot(results, aes(x=Dataset, y=chol, fill=Cluster)) + geom_bar(stat="identity")
p8=ggplot(results, aes(x=Dataset, y=fbs0, fill=Cluster)) + geom_bar(stat="identity")
p9=ggplot(results, aes(x=Dataset, y=restecg0, fill=Cluster)) + geom_bar(stat="identity")
p10=ggplot(results, aes(x=Dataset, y=thalachh, fill=Cluster)) + geom_bar(stat="identity")
p11=ggplot(results, aes(x=Dataset, y=exng1, fill=Cluster)) + geom_bar(stat="identity")
p12=ggplot(results, aes(x=Dataset, y=oldpeak, fill=Cluster)) + geom_bar(stat="identity")
p13=ggplot(results, aes(x=Dataset, y=slp0, fill=Cluster)) + geom_bar(stat="identity")
p14=ggplot(results, aes(x=Dataset, y=caa0, fill=Cluster)) + geom_bar(stat="identity")
p15=ggplot(results, aes(x=Dataset, y=caa1, fill=Cluster)) + geom_bar(stat="identity")
p16=ggplot(results, aes(x=Dataset, y=caa2, fill=Cluster)) + geom_bar(stat="identity")
p17=ggplot(results, aes(x=Dataset, y=caa3, fill=Cluster)) + geom_bar(stat="identity")
p18=ggplot(results, aes(x=Dataset, y=thall1, fill=Cluster)) + geom_bar(stat="identity")
p19=ggplot(results, aes(x=Dataset, y=thall3, fill=Cluster)) + geom_bar(stat="identity")
multiplot(p1,p2,p3)#,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14)
multiplot(p2,p3,p4)
multiplot(p5,p5,p7)
multiplot(p8,p9,p10)
multiplot(p11,p12,p13)
multiplot(p14,p15,p16)
multiplot(p17,p18,p19)

#Develop train an test set
set.seed(1234)
intrain <- createDataPartition(y=d$y, p=.80, list=F)

train <- d[intrain,]
test <- d[-intrain,]

rm(intrain)

tcontrol <- trainControl(method = "cv",
                         number = 3,
                         classProbs = T,
                         summaryFunction = twoClassSummary,
                         allowParallel = T
)

str(train)
str(test)

train <- train %>%
  mutate(y = ifelse(y == 1,"Yes","No"))
test <- test %>%
  mutate(y = ifelse(y == 1,"Yes","No"))


#Make output variable a factor
train$y <- as.factor(train$y)
test$y <- as.factor(test$y)


#Model 2
m1 <- train(y ~ age + sex1 + cp0 + cp1 + cp2 + trtbps + chol + exng1 + thall1 + thall3 , 
            data = train,
            method = "glm",
            family = "binomial",
            trControl = tcontrol
            ,metric = 'ROC')
#metric = "RMSE") #Use RMSC for regression problem

defaultSummary(data=data.frame(obs=train$y, pred=predict(m1, newdata=train))
               , model=m1)
# test 
defaultSummary(data=data.frame(obs=test$y, pred=predict(m1, newdata=test))
               , model=m1)

summary(m1)

predict(m1, test, type = 'prob')$Yes

str(data)

#######################################################################################################
#SHINY APP DEVELOPMENT
#######################################################################################################
options(java.parameters = "-Xmx64048m") # 64048 is 64 GB

ui <- fluidPage(setBackgroundColor(color = c("#F7FBFF", "#2171B5"),gradient = "linear",direction = "bottom"),
                headerPanel('Patient Detail - Risk of Heart Attack Calculation'), 
                sidebarLayout(
                  sidebarPanel(
                    br(),
                    selectInput(inputId = 'age', label = 'Select Patient Age', choices = as.numeric(unique(25:80)),selected = '50'),
                    br(),
                    selectInput(inputId = 'sex', label = 'Select Patient Sex', choices = as.character(c('M','F'))),
                    br(),
                    selectInput(inputId = 'cp', label = "Patient Experiencing Chest Pain?", choices = c('No','Slight','Moderate','Severe')),
                    br(),
                    textInput(inputId = 'bps', label = 'Resting Blood Pressure (systolic)', value = as.numeric(round(mean(data$trtbps),0))),
                    br(),
                    textInput(inputId = 'chol', label = 'Total Cholesterol', value = round(mean(data$chol),0)),
                    br(),
                    textInput(inputId = 'fbs', label = 'Fasting Blood Sugar', value = '120'),
                    br(),
                    selectInput(inputId = 'exng', label = 'Patient Experiencing Exercised Induced Angina?', choices = as.character(c('N','Y'))),
                    br(),
                    selectInput(inputId = 'thal', label = "Does the Patient have a Heart Defect?", choices = c('No', 'Fixed Defect','Reversable Defect')),
                    br(),
                    actionButton(inputId = "runPatient", label = "Determine Heart Attack Risk", icon=icon('chart-line')),
                    br(),
                    br(),
                    #actionButton(inputId = "showStats", label = "Show Patient Statistics", icon = icon('eye')),
                    #br(),
                    #h6('Image:'),
                    #tags$img(src='images/hasymptoms.png', height = 150, width = 150),
                    #tags$div(img(src='images/hasymptoms.png', height = 150, width = 150)),
                    #img(src = 'images/hasymptoms.png', height = 150, width = 150)
                    
                    
                  ),
                  #Tabular Creation w/ Graphical Ouputs
                  mainPanel(tabsetPanel(id = 'tabs',
                                        tabPanel(value = 'patientInfo', title ='Patient Info', icon = icon('clinic-medical'), 
                                                 br(),
                                                 br(), 
                                                 htmlOutput('text'), 
                                                 tags$head(tags$style("#text{color: #66100E; font-size: 24px; font-style: italic;}")),
                                        ),
                                        tabPanel(value = 'patientStats', title = 'Patient File', icon = icon('file-medical-alt'), 
                                                 br(),
                                                 plotOutput('cpbar') , 
                                                 plotOutput('exngbar')),
                                        tabPanel(value = 'genStats', title = 'General Statistics', icon = icon('bar-chart-o'), 
                                                 tabsetPanel(id = 'tab2', 
                                                             tabPanel(value = 'pie', title = 'Pie Chart Selection', icon = icon('heartbeat'),
                                                                      br(),
                                                                      selectInput(inputId = 'pie', label = 'Select Categorical Variable', choices = c('Sex',"Chest Pain", "Blood Sugar", 'Exercise Induced Angina')),
                                                                      br(), 
                                                                      plotOutput('piePlot')),
                                                             tabPanel(value = 'gen', title = 'General', icon = icon('table'),
                                                                      #tags$head(tags$style("#ageGenTxt{color: black; font-size: 18px; font-style: bold;}")), 
                                                                      #tags$head(tags$style("#ageBoxGenTxt{color: black; font-size: 18px; font-style: bold;}")), 
                                                                      #textOutput('ageGenTxt'), 
                                                                      br(), 
                                                                      plotOutput('ageGen'), 
                                                                      br(),
                                                                      #textOutput('ageBoxGenTxt'), 
                                                                      plotOutput('ageBoxGen'), 
                                                                      br(),
                                                                      plotOutput('cholGen'), 
                                                                      br(),
                                                                      plotOutput('otherBoxGen'), 
                                                                      br(),
                                                                      plotOutput('ageBpsGen'),
                                                                      br())
                                                             
                                                 )),
                                        tabPanel(value = 'references', title = 'References', icon = icon('keyboard'),
                                                 tags$head(tags$style("#dataSource{color: black; font-size: 18px; font-style: bold;}")), 
                                                 tags$head(tags$style("#References{color: black; font-size: 18px; font-style: bold;}")), 
                                                 br(),
                                                 textOutput('dataSource'),
                                                 br(),
                                                 textOutput('References'), 
                                        ) 
                                        
                                        
                  ),
                  #tableOutput('table'),
                  #tableOutput('plot1')
                  )
                  
                )
)

server <- function(input, output) {
  
  
  plotData = reactiveVal()
  selectedData <- reactive({
    data[,]
  })
  
  #Go to Patient Info Tab when Run Patient info is Clicked
  #observeEvent(input$runPatient, {
  #    updateTabsetPanel(session, 'tabs', selected = 'patientInfo')
  #  })
  
  ########################################################################################################  
  #Model Prediction - Patient Info Tab
  determineHARisk <- eventReactive(input$runPatient, {
    df <- data.frame(age=as.numeric(input$age), sex1= ifelse(input$sex=="M",1,0), cp0=ifelse(input$cp == "No",1,0) , cp1 = ifelse(input$cp == "Slight",1,0),
                     cp2 = ifelse(input$cp == "Moderate", 1,0) , trtbps = as.numeric(input$bps) , fbs0 = ifelse(input$fbs > 120, 1,0) , chol = as.numeric(input$chol),
                     exng1 = ifelse(input$exng == 'Y',1,0), thall1 = ifelse(input$thal == 'No',1,0) , thall3 = ifelse(input$thal=='Reversable Defect',1,0))
    waist <- ifelse(input$sex == 'M', 37,35)
    alcohol <-ifelse(input$sex == 'M', "two", "one")
    if(round(predict(m1, df, type = 'prob')$Yes,3) < 0.250){
      mhl <- "low"
    }
    else if (round(predict(m1, df, type = 'prob')$Yes,3) < 0.667){
      mhl <- "moderate"
    }
    else {
      mhl <- "high"
    }
    mbrProb <- paste0("The liklihood of a heart attack is ", mhl,": ", round(predict(m1, df, type = 'prob')$Yes,3)*100,"%")
    
    mbrStment <- if(!mhl=="low"){
      HTML(mbrProb, '<br><br><br><br>', 'Here are few ways to reduce the risk of a Heart Attack: <br><br>', 
           '1.) There is risk reduction attributed to not smoking. <br><br>',
           '2.) Diet that is rich in fruits, vegetables, legumes, nuts, reduced-fat dairy products, and fish. <br><br>',
           '3.) Maintain a waistline of ' , waist, ' inches or less. <br><br>',
           '4.) Drink fewer than ', alcohol,' alcohol beverages per day. <br><br>',
           '5.) Moderate daily and/or weekly exercise. <br><br>',
           '6.) Maintain a habit of all of the above.* <br><br>')
    }
    else {
      HTML(mbrProb)
    }
    
  })
  
  output$text <- renderText({
    determineHARisk()
  })
  
  
  
  
  ################################################################################################
  # Patient Statistics Tab
  
  #Chest Pain
  output$angTxt <- renderText({
    if(!input$cp == 'No'){
      paste0("Impact of Chest Pain on ",ifelse(input$sex == "M", "males","females")," on Heart Attacks")
    }
  })
  output$cpbar = renderPlot({
    if(!input$cp == 'No' & input$sex == "M"){
      cpbar <- ggplot(data[data$sex=="1",], aes(cp)) + 
        geom_bar(aes(fill=output),position = "fill") +
        xlab("Chest Pain Severity") +
        ylab("Frequency") + 
        labs(fill = '') +
        ggtitle("Impact of Chest Pain on Heart Attacks") +
        scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c("Normal", "Heart Attack")) +
        theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic")) +
        scale_x_discrete(labels = c('No','Slight','Moderate','Severe'))
      
      plot(cpbar)
    }
    if(!input$cp == 'No' & input$sex == "F"){
      cpbar <- ggplot(data[data$sex=="0",], aes(cp)) + 
        geom_bar(aes(fill=output),position = "fill") +
        xlab("Chest Pain Severity") +
        ylab("Frequency") + 
        labs(fill = '') +
        ggtitle("Impact of Chest Pain on Heart Attacks") +
        scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c("Normal", "Heart Attack")) +
        theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic")) +
        scale_x_discrete(labels = c('No','Slight','Moderate','Severe'))
      
      plot(cpbar)
    }
  })
  
  #Angina
  
  #output$angTxt <- renderText({
  #  if(!input$exng == 'N'){
  #    paste0("Impact of Exercised Induced Angina on ",ifelse(input$sex == "M", "males","females")," on Heart Attacks")
  #  }
  #})
  
  
  output$exngbar = renderPlot({
    if(!input$exng == 'N' & input$sex == "M"){
      exngbar <- ggplot(data[data$sex=="1",], aes(exng)) + 
        geom_bar(aes(fill=output),position = "fill") +
        xlab("Exercise Induced Angina") +
        ylab("Frequency") + 
        labs(fill = '') +
        ggtitle("Impact of Exercise Induced Angina on Heart Attacks") +
        scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c("Normal", "Heart Attack")) +
        theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic")) +
        scale_x_discrete(labels = c('No','Yes'))
      
      plot(exngbar)
    }
    if(!input$exng == 'N' & input$sex == "F"){
      exngbar <- ggplot(data[data$sex=="0",], aes(exng)) + 
        geom_bar(aes(fill=output),position = "fill") +
        xlab("Exercise Induced Angina") +
        ylab("Frequency") + 
        labs(fill = '') +
        ggtitle("Impact of Exercise Induced Angina on Heart Attacks") +
        scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c("Normal", "Heart Attack")) +
        theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic")) +
        scale_x_discrete(labels = c('No','Yes'))
      
      plot(exngbar)
    }
  })
  
  ################################################################################################
  #PIE CHARTS
  
  
  output$piePlot = renderPlot({
    
    if(input$pie == "Sex"){
      pieChart <- ggplot(data, aes(x = '', y = sex, fill = sex)) +
        geom_bar(stat = 'identity', width = 1) + 
        coord_polar("y", start = 0) +
        labs(fill = '') +
        theme_void() + 
        #geom_text(aes(x=1.6, label = scales::percent(data1$pct, accuracy = 0.1)), position = position_stack(vjust = 0.5)) +
        scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c('Female','Male')) +
        #scale_fill_brewer(palette = "Blues") +
        ggtitle("Distribution of Sex in Dataset") + 
        theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic")) 
      
      plot(pieChart)
    }
    
    if(input$pie == "Chest Pain"){
      pieChart <- ggplot(data, aes(x = '', y = cp, fill = cp)) +
        geom_bar(stat = 'identity', width = 1) + 
        coord_polar("y", start = 0) +
        labs(fill = '') +
        theme_void() + 
        scale_fill_manual(values = c("#CEB888", "#66100E","#373A35", "#C28E0E"), labels = c('No','Slight','Moderate','Severe')) +
        #scale_fill_brewer(palette = "Blues") +
        ggtitle("Distribution of Chest Pain in Dataset") + 
        theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic")) 
      
      plot(pieChart)
    }
    
    if(input$pie == "Blood Sugar"){
      pieChart <- ggplot(data, aes(x = '', y = fbs, fill = fbs)) +
        geom_bar(stat = 'identity', width = 1) + 
        coord_polar("y", start = 0) +
        labs(fill = '') +
        theme_void() + 
        scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c('< 120','120 +')) +
        #scale_fill_brewer(palette = "Blues") +
        ggtitle("Distribution of Fasting Blood Sugar in Dataset") + 
        theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic")) 
      
      plot(pieChart)
    }
    
    if(input$pie == "Exercise Induced Angina"){
      pieChart <- ggplot(data, aes(x = '', y = exng, fill = exng)) +
        geom_bar(stat = 'identity', width = 1) + 
        coord_polar("y", start = 0) +
        labs(fill = '') +
        theme_void() + 
        scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c('No','Yes')) +
        #scale_fill_brewer(palette = "Blues") +
        ggtitle("Distribution of Exercise Induce Angina in Dataset") + 
        theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic")) 
      
      plot(pieChart)
    }
    
  }) 
  
  
  ################################################################################################
  #General Statistics Tab
  
  #output$ageGenTxt <- renderText({
  #  paste0("Distribution of Population Age (Normal & Heart Attack)")
  #})
  
  
  #Age Distribution Graph - General Statistics Tab
  output$ageGen = renderPlot({
    age_density <- ggplot(data=data, aes(x=age, group = output)) + geom_density(aes(fill=output), alpha = 0.75) + 
      xlab("Age") +
      ylab("Density") + 
      ggtitle("Distribution of Age") +
      labs(fill = "") +
      scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c("Normal", "Heart Attack")) +
      theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic"))
    
    plot(age_density)
    
  })
  
  ##Age BoxPlot Dist
  #output$ageBoxGenTxt <- renderText({
  #  paste0("Distribution of Population Age (BoxPlot)")
  #})
  
  #Age Boxplot
  output$ageBoxGen = renderPlot({
    ageBoxGen <- ggplot(data=data, aes(x=output, y=age)) + 
      geom_boxplot(aes(fill=output)) +      
      xlab("Normal / Heart Attack") +
      ylab("Age") + 
      ggtitle("Distribution of Age (Box Plot)") +
      scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c("Normal", "Heart Attack")) +
      theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic"),axis.ticks.x = element_blank(), axis.text.x = element_blank())
    
    plot(ageBoxGen)
  })
  
  #Cholesterol Distribution - General Stats Tab
  output$cholGen = renderPlot({
    chol_density <- ggplot(data=data, aes(x=chol, group = output)) + geom_density(aes(fill=output), alpha = 0.75) + 
      xlab("Cholesterol") +
      ylab("Density") + 
      ggtitle("Distribution of Cholesteral") +
      labs(fill = "") +
      scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c("Normal", "Heart Attack")) +
      theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic"))
    
    plot(chol_density)
  })
  
  #cholesterol Boxplot by Sex
  output$otherBoxGen = renderPlot({
    data2 <- data
    data2$sex <- revalue(x=data2$sex, c('0' = 'Female','1'='Male'))
    sex_names <- list("Female" = '0',"Male" = '1')
    
    otherBoxGen <- ggplot(data=data2, aes(x=output, y=chol)) + geom_boxplot(aes(fill=output)) +      
      xlab("Normal / Heart Attack") +
      ylab("Cholesterol") + 
      ggtitle("BoxPlot of Cholesterol by Sex ") +
      labs(fill = "") + 
      scale_fill_manual(values = c("#CEB888", "#66100E"), labels = c("Normal", "Heart Attack")) +
      theme(plot.title = element_text(color = "black", size = 18, face = "bold.italic"),axis.ticks.x = element_blank(), axis.text.x = element_blank()) +
      facet_wrap(~ sex, ncol = 2, labeller = as_labeller(sex_names)) 
    
    plot(otherBoxGen)
  })
  
  #Age/Blood Pressure Scatter
  output$ageBpsGen = renderPlot({
    data2 <- data
    data2$sex <- revalue(x=data2$sex, c('0' = 'Female','1'='Male'))
    sex_names <- list("Female" = '0',"Male" = '1')
    ageBpsGen <- ggplot(data=data2, aes(x=trtbps, y=age, group = output)) + 
      geom_point(aes(color = output)) +      
      xlab("Blood Pressure") +
      ylab("Age") + 
      labs(color = "") +
      ggtitle("ScatterPlot of Age/Blood Pressure by Sex") +
      scale_color_manual(values = c("#CEB888", "#66100E"), labels = c("Normal", "Heart Attack")) +
      theme(plot.title = element_text(color = 'black', size = 18, face = "bold.italic")) +
      facet_wrap( ~ sex, ncol = 2, labeller = as_labeller(sex_names)) +
      geom_smooth(aes(fill = output), method = "lm", size = 0.5, alpha = 0.2) +
      scale_fill_manual(values = c('#CEB888','#66100E'), labels = c("Normal", "Heart Attack")) +
      labs(fill = "")
    
    
    plot(ageBpsGen)
  })
  
  
  ################################################################################################ 
  #REFERENCES
  output$dataSource <- renderText({
    paste0("Dataset Location: https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset")
  })
  output$References <- renderText({
    paste0("*Reference Location: https://heart.arizona.edu/heart-health/prevention/five-ways-reduce-heart-attack-risk-80-percent")
  })
  
  
  
  ################################################################################################ 
  
}# end of server


shinyApp(ui = ui, server = server)