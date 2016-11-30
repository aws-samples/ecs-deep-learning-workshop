# Deploy a Deep Learning Framework on Amazon ECS: Lab Guide
  
  
## Overview:
[Deep Learning (DL)](https://en.wikipedia.org/wiki/Deep_learning) is an implementation of [Machine Learning (ML)](https://en.wikipedia.org/wiki/Machine_learning) that uses neural networks to solve difficult problems such as image recognition, sentiment analysis and recommendations.  Neural networks simulate the functions of the brain where artificial neurons work in concert to detect patterns in data.  This allows deep learning algorithms to classify, predict and recommend with a high degree of accuracy as more data is trained in the network.  DL algorithms generally operate with a high degree of parallelism and is computationally intense.  As a result, emerging deep learning libraries, frameworks, and platforms allow for data and model parallelization and can leverage advancements in GPU technology for improved performance.  
This workshop will walk you through the deployment of a deep learning library called [MXNet](http://mxnet.io) on AWS using [Docker containers](https://www.docker.com/).  Containers provide isolation, portability and repeatability, so your developers can easily spin up an environment and start building without the heavy lifting.  

The goal is not to go deep on the learning (no pun intended) aspects, but to illustrate how easy it is to deploy your deep learning environment on AWS and use the same tools to scale your resources as needed.  

### Requirements:  
* AWS account - if you don't have one, it's easy and free to [create one](https://aws.amazon.com/)
* AWS IAM account with elevated privileges allowing you to interact with CloudFormation, IAM, EC2, ECS, ECR, and CloudWatch Logs
* A workstation or laptop with an ssh client installed, such as [putty](http://www.putty.org/)
* Familiarity with python, Docker, AWS, and machine learning - not required but a bonus

### Labs:  
These labs are designed to be completed in sequence.  If you are reading this at a live AWS event, the workshop attendants will give you a high level run down of the labs.  Then it's up to you to follow the instructions below to complete the labs.  Don't worry if you're embarking on this journey in the comfort of your office or home- presentation materials can be found in the git repo in the top-level [presentations](https://github.com/awslabs/ecs-deep-learning-workshop/tree/master/presentations) folder.

**Lab 1:** Setup the workshop environment on AWS  
**Lab 2:** Build an MXNet Docker Image  
**Lab 3:** Deploy the MXNet Container with ECS  
**Lab 4:** Image Classification with MXNet  

### Conventions:  
Throughout this README, we provide commands for you to run in the terminal.  These commands will look like this: 

<pre>
$ ssh -i <b><i>private_key.pem</i></b> ec2-user@<b><i>ec2_public_DNS_name</i></b>
</pre>

The command starts after $.  Words that are ***bolded italics*** indicate a value that is unique to your environment.  For example, the ***private\_key.pem*** refers to the private key of an SSH key pair that you've created, and the ***ec2\_public\_DNS\_name*** is a value that is specific to an EC2 instance launched in your account.  

### Workshop Cleanup:
This section will appear again below as a reminder because you will be deploying infrastructure on AWS which will have an associated cost.  Fortunately, this workshop should take no more than 2 hours to complete, so costs will be minimal.  See the appendix for an estimate of what this workshop should cost to run.  When you're done with the workshop, follow these steps to make sure everything is cleaned up.  

1. Delete any manually created resources throughout the labs. 
2. Delete any data files stored on S3 and container images stored in ECR.  
3. Delete the CloudFormation stack launched at the beginning of the workshop. 
	
* * * 

## Let's Begin!  

### Your Challenge:  
There are just not enough cat pictures on social media these days, to the point where it would be amazing to have a social network dedicated to devoted cat lovers around the world.  The problem is, how do you make sure images uploaded to this niche network are cat related?  Image classification to the rescue!  

Implement MXNet to recognize a variety of images, so you can specifically identify ones of our favorite feline friend!   

Here is the overall architecture of what you will be building throughout this workshop.  By the end of the workshop, you will have the ability to interact directly with the MXNet containers using SSH or Jupyter notebooks.  You will also have the option to create Tasks in ECS which can be run through the management console, CLI or SDKs. 

![Overall Architecture](/images/architecture.png)

### Lab 1 - Set up the Workshop Environment on AWS:    

1\. First you'll need to create an SSH key pair which will be used to login to the instances once provisioned.  Go to the EC2 Dashboard and click on **Key Pairs** in the left menu under Network & Security.  Click **Create Key Pair**, provide a name (can be anything, make it something memorable) when prompted, and click **Create**.  Once created, the private key in the form of .pem file will be automatically downloaded.    

2\. For your convenience, we provide a CloudFormation template to stand up the core infrastructure.  

*Prior to launching a stack, be aware that a few of the resources launched need to be manually deleted when the workshop is over. When finished working, please review the "Workshop Cleanup" section to learn what manual teardown is required by you.*

Click on one of these CloudFormation templates to launch your stack:  

Region | Launch Template
------------ | -------------  
**Ohio** (us-east-2) | [![Launch ECS Deep Learning Stack into Ohio with CloudFormation](/images/deploy-to-aws.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-2#/stacks/new?stackName=ecs-deep-learning-stack&templateURL=https://s3.amazonaws.com/ecs-dl-workshop-us-east-2/ecs-deep-learning-workshop.yaml)  
**Oregon** (us-west-2) | [![Launch ECS Deep Learning Stack into Oregon with CloudFormation](/images/deploy-to-aws.png)](https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=ecs-deep-learning-stack&templateURL=https://s3.amazonaws.com/ecs-dl-workshop-us-west-2/ecs-deep-learning-workshop.yaml)  

The template will automatically bring you to the CloudFormation Dashboard and start the stack creation process in the specified region.  The template sets up a VPC, IAM roles, S3 bucket, ECR container registry, and an ECS cluster which is comprised of two EC2 instances with the Docker daemon running on each.  In order to keep costs low in the workshop, the EC2 instances are spot instance deployed by Spot Fleet.  The idea is to provide a contained environment, so as not to interfere with any other things in your account.  If you are new to CloudFormation, take the opportunity to review the template during stack creation.  

**Checkpoint**  
Periodically check on the stack creation process in the CloudFormation Dashboard.  If all goes well, your stack should show status CREATE\_COMPLETE.  In the Outputs tab, take note of the **ecrRepository** and **spotFleetName** values; you will need these in the next lab.     

![CloudFormation CREATION\_COMPLETE](/images/cf-complete.png)

If there was an error during the stack creation process, CloudFormation will rollback and terminate.  You can investigate and troubleshoot by looking in the Events tab.  Any errors encountered during stack creation will appear in the event log.      

### Lab 2 - Build an MXNet Docker Image:    
In this lab, you will build an MXNet docker image using one of the ECS cluster instances which already comes bundled with Docker installed.  There are quite a few dependencies for MXNet, so for your convenience, we provide a Dockerfile in the lab 2 folder to make sure nothing is missed.  MXNet uses SSH as the mechanism for communication between containers, so you'll be generating an SSH key pair to configure public key authentication for secure access.  You can review the Dockerfile to see what's being installed.  Links to MXNet documentation can be found in the Appendix if you'd like to read more about it.  

1\. Go to the EC2 Dashboard in the Management Console and click on **Instances** in the left menu.  Select one of the two EC2 instances created by the CloudFormation stack.  If your instances list is cluttered with other instances, apply a filter in the search bar using the tag key **aws:ec2spot:fleet-request-id** and choose the value that matches the **spotFleetName** from your CloudFormation Outputs.  

![EC2 Public DNS](/images/ec2-public-dns.png)

Once you've selected one of the provisioned EC2 instances, note the Public DNS Name and SSH into the instance.  

<pre>
$ ssh -i <b><i>private_key.pem</i></b> ec2-user@<b><i>ec2_public_DNS_name</i></b>
</pre>

2\. Once logged into the EC2 instance, clone the workshop github repository so you can easily access the Dockerfile.  
`$ git clone https://github.com/awslabs/ecs-deep-learning-workshop.git`  

3\. Navigate to the lab-2-build/mxnet/ folder to use as our working directory.  
`$ cd ecs-deep-learning-workshop/lab-2-build/mxnet`

4\. Build the Docker image using the provided Dockerfile.

`$ docker build -t mxnet .`  

This process will take about 10-15 minutes because MXNet is being compiled during the build process.  If you're new to Docker, you can take this opportunity to review the Dockerfile to understand what's going on or take a quick break to grab some coffee/tea.  

5\. One of the goals of the workshop is to provide an interactive Jupyter notebook for your teams to work with MXNet.  Jupyter runs as a web application on your container and by default does not require a password.  You can add a hashed password to the Jupyter config file to enable one.  You can make this change in an interactive bash shell and commit the change to your local image.  First start the interactive session.

`$ docker run -ti mxnet /bin/bash`

You'll notice your prompt has changed to: <pre><i>root@<b>2b3b44bd0eed</b>:~/mxnet#</i></pre>
The bolded portion will be unique and represents your container ID.  Note this down because you'll need it later when you want to commit your changes.  

6\. First let's generate a hashed password using the passwd() method provided by Jupyter.  Start python in interactive mode and run the passwd function to generate the hashed password.  **Note:** Commands are bolded below, and you'll be prompted to enter your password once to set it and again to confirm it.  The output will be a sha1 hash.  Note this down because you'll be adding this to the Jupyter config.  

<pre>
root@2b3b44bd0eed:~/mxnet# <b>python</b>
Python 2.7.6 (default, Jun 22 2015, 17:58:13) 
[GCC 4.8.2] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> <b>from notebook.auth import passwd</b>
>>> <b>passwd()</b>
Enter password: 
Verify password: 
'sha1:51544180f6e8:4c080dd3fcd90736e21903fbef215a45df63f851'
>>> <b>exit()</b>
</pre>  

7\. Use vim to edit ~/.jupyter/jupyter_notebook_config.py. Find *#c.NotebookApp.password = u''*, delete the leading "#", and insert the sha1 hashed password between the single quotes.  It should look like this when you're done (note, the sha1 below is an example, and you would insert your sha1 hash):  

`c.NotebookApp.password = u'sha1:51544180f6e8:4c080dd3fcd90736e21903fbef215a45df63f851'`  

Once you've made that change, save and close the file.  If vim is not your editor of choice, you can install your preferred editor, e.g. nano, emacs, to edit the file above.   

8\. You can now exit your interacive container session:

`root@2b3b44bd0eed:~/mxnet# exit`  

8\. At this point, you've edited the container, and in order for your changes persist in the image, you need to commit the changes.  You can do this by using the docker commit command like so: 

`$ docker commit -m "added password for Jupyter notebook" -a "John Smith" 2b3b44bd0eed mxnet`

The command specifies a "-m" flag which is a commit message and a "-a" flag which indicates the author of the change.  You're also passing in the unique container ID and the image that you'd like to commit the changes to.  

9\. Now that you've committed the change you made to your local docker image, tag and push the MXNet Docker image to ECR.  You'll reference this image when you deploy the container using ECS in the next lab.  Below is the command and format for the repository URI.  You can find your respository URI in the EC2 Container Service Dashboard; click on **Repositories** in the left menu and click on the repository name that matches the **ecrRepository** output from CloudFormation. The Repository URI will be listed at the top of the screen.  

![ECR URI](/images/ecr-uri.png)  

<pre>
$ docker tag mxnet:latest <b><i>aws_account_id</i></b>.dkr.ecr.<b><i>region</i></b>.amazonaws.com/<b><i>ecr_repository</i></b>:latest   
$ docker push <b><i>aws_account_id</i></b>.dkr.ecr.<b><i>region</i></b>.amazonaws.com/<b><i>ecr_repository</i></b>:latest  
</pre>

**Checkpoint**  
At this point you should have a working MXNet Docker image to deploy with ECS.  If you don't see errors following the commands above, you should be in good shape.  


### Lab 3 - Deploy the MXNet Container with ECS:    
Now that you have an MXNet image ready to go, next step is to create a task definition, which specifies parameters used by ECS to run your container, e.g. Docker image, cpu/memory resource requirements, host:container port mappings.  You'll notice that the params in the task definition closely match options passed to a Docker run command.  Task definitions can be used to deploy multiple containers that are linked together, for example, an application server and database.  In this workshop, we will focus on deploying a single container.         

1\. Open the EC2 Container Service dashboard, click on **Task Definitions** in the left menu, and click **Create new Task Definition**.    

*Note: You'll notice there's a task definition already there in the list.  Ignore this until you reach lab 5.*  

2\. First, name your task definition, e.g. "mxnet".  If you happen to create a task definition that is a duplicate of an existing task definition, ECS will create a new revision, incrementing the version number automatically.  

3\. Next click on **Add container** and complete the Standard fields in the Add container window.  Provide a name for your container, e.g. "mxnet", which is functionally equivalent to the --name option of the Docker run command. This name value can also be used for any container linking.  The image field is the container image that you will be deploying.  The format is equivalent to the *registry/repository:tag* format used in lab 2, step 6, i.e. ***aws_account_id***.dkr.ecr.***region***.amazonaws.com/***ecr_repository***:latest.  

Finallly, set the memory to "2048" and map the host port 80 to the container port 8888.  Port 8888 is the listening port for the Jupter notebook configuration, and we map it to port 80 to reduce running into issues with proxies during the workshop.  You can leave all other fields as default.  Click **Add** to save this configuration and add it to the task defintion.  Click **Create** to complete the task defintion creation step.         

![Task Definition](/images/task-def.png)  

4\. Now that you have a task definition created, you can have ECS deploy an MXNet container to your EC2 cluster using the Run Task option.  On the screen, click on the **Actions** dropdown menu and select **Run Task**.  Choose your ECS Cluster from the dropdown menu.  If you have multiple ECS Clusters in the list, you can find your workshop cluster by referring to the **ecsClusterName** value from the CloudFormation stack Outputs tab.  Keep number of tasks set to 1 and click on **Run Task**.  ECS is now running your MXNet container on an ECS cluster instance with available resources.  If you run multiple tasks, ECS will balance out the tasks across the cluster, so one cluster instance doesn't have a disproportionate number of tasks.  

5\. On the Clusters page, you'll see a Tasks tab towards the bottom of the page.  Notice your new task starts in the Pending state.  Click on the refresh button after a little bit to refresh the contents of that tab, and once it is in the Running state, you can test accessing the Jupyter notebook.  In addition to the running state, this tab also identifies which Container Instance the task is running on.  Click on the Container Instance and you'll see the Public DNS of the EC2 instance on the next page.   

![Run Task](/images/task-run.png)  

6\. Open a new web browser tab and load the public DNS name - http://***ec2_public_DNS_name***.  You should see the Jupyter login page.

![Jupyter Login](/images/jupyter-login.png)  

### Lab 4 - Image Classification with MXNet:   
Now that you have an MXNet container built and deployed with ECS, you can try out an image classification example provided by MXNet to make sure the framework is working properly.  There are two examples you can run through, one for training a model and one for generating a prediction.  Both examples are presented in the form of a Jupyter notebook.  You may have noticed that Jupyter was installed and configured during the creation of the MXNet image.  If you're new to Jupyter, it is essentially a web application that allows you to interactively step through blocks of written code.  The code can be edited by the user as needed or desired, and there is a play button that lets you step through the cells.  Cells that do not code have no effect, so you can hit play to pass through the cell.          

#### Training:    
In the MXNet example for training an Mnist model, there is a python file that runs the training. You will SSH into the same host that already has Jupyter running that you found in step 5 of lab 3, connect to a specific container, and finally run the training command.

First, SSH into the instance:
<pre>
$ ssh -i <b><i>private_key.pem</i></b> ec2-user@<b><i>ec2_public_DNS_name</i></b>
</pre>

Once logged in, find the container to connect to by running:
<pre>
$ docker ps
</pre>

On the left hand side, you'll find two containers that are running. One for our mxnet container, and one for the amazon-ecs-agent. Note down the CONTAINER_ID of the mxnet image as connect into this specific container. To drop into a shell, execute the /bin/bash command like this:

<pre>
$ docker exec -it <b><i>CONTAINER_ID</i></b> /bin/bash
</pre>

Now that you're in the container, you can feel free to navigate around. It should look very similar compared to what you saw in lab 2. Once you're ready, navigate to /root/ecs-deep-learning-workshop/mxnet/example/image-classification/ and run train_mnist.py

<pre>
$ cd /root/ecs-deep-learning-workshop/mxnet/example/image-classification/
$ python train_mnist.py
</pre>

You will start to see output right away. It will look like:
<pre>
INFO:root:Start training with [cpu(0)]
INFO:root:Epoch[0] Batch [100]	Speed: 13736.09 samples/sec	Train-accuracy=0.782969
INFO:root:Epoch[0] Batch [200]	Speed: 12799.08 samples/sec	Train-accuracy=0.910000
INFO:root:Epoch[0] Batch [300]	Speed: 13594.84 samples/sec	Train-accuracy=0.926094
INFO:root:Epoch[0] Batch [400]	Speed: 13775.83 samples/sec	Train-accuracy=0.933594
INFO:root:Epoch[0] Batch [500]	Speed: 13732.46 samples/sec	Train-accuracy=0.937656
INFO:root:Epoch[0] Batch [600]	Speed: 13788.14 samples/sec	Train-accuracy=0.941719
INFO:root:Epoch[0] Batch [700]	Speed: 13735.79 samples/sec	Train-accuracy=0.937813
INFO:root:Epoch[0] Batch [800]	Speed: 13789.07 samples/sec	Train-accuracy=0.944531
INFO:root:Epoch[0] Batch [900]	Speed: 13754.83 samples/sec	Train-accuracy=0.953750
</pre>

As you should be able to tell, logging into a machine, then dropping into a shell onto a container isn't the best process to do all of this, and it's very manual. For the prediction section, we will show you a more UI based interactive way of running some commands.


#### Prediction:    
Since training a model can be resource intensive and a lengthy process, you will run through an example that uses a pre-trained model built from the full [ImageNet](http://image-net.org/) dataset, which is a collection of over 10 million images with thousands of classes for those images.  This example is built with a Juypter notebook, so you can interactively walk through the example. 

1\. Open a web browser and visit this URL to access the Jupyter notebook for the demo; the password is what you configured when you built the image in lab 2.  
http://***ec2_public_dns_name***/notebooks/mxnet-notebooks/python/tutorials/predict_imagenet.ipynb

2\. Play through the cells to run through this example, which loads and prepares the pre-trained model as well as provide methods to load images into the model to predict its classification.  If you've never used Jupyter before, you're probably wonder how you know something is happening.  Cells with code are denoted on the left with "In [n]" where n is simply a cell number.  When you play a cell that requires processing time, the number will show an asterisk.  

**IMPORTANT**: In cell 2, the default context is to use gpu, but in the case of this workshop, we're using cpu resources so change the text "gpu" to "cpu".  Being able to switch between using cpu and gpu is a great feature of this library.  See the following screenshot which illustrates where to change from gpu to cpu; also highlighted in the screenshot is the play button which lets you run the cells.  While deep learning performance is better on gpu, you can make use of cpu resources in dev/test environments to keep costs down.  

![Jupyter Notebook - Prediction](/images/jupyter-notebook-predict.png)

3\. Once you've stepped through the two examples at the end of the notebook, try feeding arbitrary images to see how well the model performs.  Remember that Jupyter notebooks let you input your own code in a cell and run it, so feel free to experiment.    

### Lab 5 - Wrap Image Classification in an ECS Task:
At this point, you've run through training and prediction examples using the command line and using a Juypter notebook, respectively.  You can also create task definitions to execute these commands, log the outputs to both S3 and CloudWatch Logs, and terminate the container when the task has completed.  S3 will store a log file containing the outputs from each task run, and CloudWatch Logs will have a log group that continues to append outputs from each task run.  In this lab, you will create additional task definitions to accomplish this.  The steps should be familiar because you've done this in lab 3, only the configuration of the task definition will be slightly different. 

*Note: The task definition that was created by the CloudFormation template is an example of a prediction task that you can refer to for help if you get stuck.*  


#### Training task

1\. Open the EC2 Container Service dashboard, click on **Task Definitions** in the left menu, and click **Create new Task Definition**.    

2\. Name your task definition, e.g. "mxnet-train".  

3\. Click on **Add container** and complete the Standard fields in the Add container window.  Provide a name for your container, e.g. "mxnet-train".  The image field is the same container image that you deployed previously.  As a reminder, the format is equivalent to the *registry/repository:tag* format used in lab 2, step 6, i.e. ***aws_account_id***.dkr.ecr.***region***.amazonaws.com/***ecr_repository***:latest.  

Set the memory to "2048".  Leave the port mapping blank because you will not be starting the Jupyter process, and instead running a command to perform the training.  

Scroll down to the **Advanced Container configuration** section, and in the **Entry point** field, type:  

`/bin/bash, -c`

In the **Command** field, type:  

<pre>
DATE=`date -Iseconds` && echo \\\"running train_mnist.py\\\" && cd /root/ecs-deep-learning-workshop/mxnet/example/image-classification/ && python train_mnist.py | tee results && echo \\\"results being written to s3://$OUTPUTBUCKET/train_mnist.results.$HOSTNAME.$DATE.txt\\\" && aws s3 cp results s3://$OUTPUTBUCKET/train_mnist.results.$HOSTNAME.$DATE.txt && echo \\\"Task complete!\\\"
</pre>  

The command references an OUTPUTBUCKET environment variable, and you can set this in **Env variables**.  Set the key to be "OUTPUTBUCKET" and the value to be the S3 output bucket created by CloudFormation.  You can find the value of your S3 output bucket by going to the CloudFormation stack outputs tab, and used the value for **outputBucketName**.    

![Advanced Configuration - Environment](/images/adv-config-env-train.png)  

Next you'll configure logging to CloudWatch Logs.  Scroll down to the **Log configuration**, select **awslogs** from the *Log driver* dropdown menu.  For *Log options*, set the **awslogs-group** to be the value of **cloudWatchLogsGroupName** from the CloudFormation stack outputs tab.  And type in the region you're currently using, e.g. Ohio would be us-east-2, Oregon would be us-west-2.  Leave the **awslogs-stream-prefix** blank.  

![Advanced Configuration - Log configuration](/images/adv-config-log-train.png)  
  
If you are using GPU instances, you will need to check the box for **Privileged** in the **Security** section.  Since we're using CPU instances, leave the box unchecked.          

Click **Add** to save this configuration and add it to the task defintion.  Click **Create** to complete the task defintion creation step.         

4\. Now you're ready to test your task definition.  Select **Run Task** from the **Actions** drop down.  Refresh the task list to confirm the task enters the Running state.  

5\. The task outputs logs to CloudWatch Logs as well as S3.  Open the **CloudWatch** dashboard, and click on **Logs** in the left menu.  Click on the log group, and then click on the log stream that was created.  You should see log output from the task run; since the training task takes some time to complete, you'll see the log output continue to stream in.  Once the task has completed and stopped, check your S3 output bucket, and you should see a log file has been written.  Download the log file and check the content.

![CloudWatch Logs](/images/cw-logs.png)  


#### Prediction task

1\. Return to the **Task Definitions** page, and click **Create new Task Definition**.    

2\. Name your task definition, e.g. "mxnet-predict".  

3\. Click on **Add container** and complete the Standard fields in the Add container window.  Provide a name for your container, e.g. "mxnet-predict".  The image field is the same container image that you deployed previously.  As a reminder, the format is equivalent to the *registry/repository:tag* format used in lab 2, step 6, i.e. ***aws_account_id***.dkr.ecr.***region***.amazonaws.com/***ecr_repository***:latest.  

Set the memory to "2048".  Leave the port mapping blank because you will not be starting the Jupyter process, and instead running a command to perform the training.  

Scroll down to the **Advanced Container configuration** section, and in the **Entry point** field, type:  

`/bin/bash, -c`

In the **Command** field, type:  

<pre>
DATE=`date -Iseconds` && echo \"running predict_imagenet.py $IMAGEURL\" && /usr/local/bin/predict_imagenet.py $IMAGEURL | tee results && echo \"results being written to s3://$OUTPUTBUCKET/predict_imagenet.results.$HOSTNAME.$DATE.txt\" && aws s3 cp results s3://$OUTPUTBUCKET/predict_imagenet.results.$HOSTNAME.$DATE.txt && echo \"Task complete!\"
</pre>  

Similar to the training task, configure the **Env variables** used by the command.  Set "OUTPUTBUCKET" to be the value of **outputBucketName** from the CloudFormation stack outputs tab.  Set "IMAGEURL" to be an URL to an image to be classified.  This can be a URL to any image, but make sure it's an absolute path to an image file and not one that is dynamically generated.      

![Advanced Configuration - Environment](/images/adv-config-env-predict.png)  

Configure the **Log configuration** section as you did for the training task.  Select **awslogs** from the *Log driver* dropdown menu.  For *Log options*, set the **awslogs-group** to be the value of **cloudWatchLogsGroupName** from the CloudFormation stack outputs tab.  And type in the region you're currently using, e.g. Ohio would be us-east-2, Oregon would be us-west-2.  Leave the **awslogs-stream-prefix** blank.    
  
If you are using GPU instances, you will need to check the box for **Privileged** in the **Security** section.  Since we're using CPU instances, leave the box unchecked.          

Click **Add** to save this configuration and add it to the task defintion.  Click **Create** to complete the task defintion creation step. 

4\. Run the predict task and check both CloudWatch Logs and the S3 output bucket for related log output.  


### Extra Credit Challenges:
* An S3 input bucket was created by the CloudFormation template.  Try uploading images to S3 and running the prediction task against those images.
* Trigger a lambda function when an image is uploaded to the S3 input bucket and have that lambda function call the prediction task.  

* * *

## Finished!  
Congratulations on completing the lab...*or at least giving it a good go*!  This is the workshop's permananent home, so feel free to revisit as often as you'd like.  In typical Amazon fashion, we'll be listening to your feedback and iterating to make it better.  If you have feedback, we're all ears!  Make sure you clean up after the workshop, so you don't have any unexpected charges on your next monthly bill.  

* * *

## Workshop Cleanup:

1. Delete any manually created resources throughout the labs.
2. Delete any data files stored on S3 and container images stored in ECR.
3. Delete the CloudFormation stack launched at the beginning of the workshop.

* * *

## Appendix:  

### Estimated Costs:    
Here are estimated costs for running this 2 hour workshop.    
*add cost estimate for EC2 and other*  

### Learning Resources:  
Here are additional resources to learn more about AWS, Docker, MXNet.  

* [Amazon Web Services](https://aws.amazon.com/)  
* [A Cloud Guru - online self-paced labs](https://acloud.guru/courses)  
* [Docker documentation](https://docs.docker.com/)  
* [MXNet](http://mxnet.io/)  
* [MXNet Examples](http://mxnet.io/tutorials/index.html)  

#### Articles:  
* [Powering ECS Clusters with Spot Fleet](https://aws.amazon.com/blogs/compute/powering-your-amazon-ecs-clusters-with-spot-fleet/)  
* [Distributed Deep Learning Made Easy](https://aws.amazon.com/blogs/compute/distributed-deep-learning-made-easy/)

*add links to ECS sessions*  
*add link to Werner's email about MXNet and Amazon*
*add link to MAC401 session*  



