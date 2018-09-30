import os
path = '(change this to dataset path)'
i = 0
for filename in os.listdir(path):
    os.rename(os.path.join(path,filename), os.path.join(path,'(change this to class name).'+str(i)+'.jpg'))
    i = i +1
