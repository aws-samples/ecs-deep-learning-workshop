# Deploy a Deep Learning Framework on Amazon ECS: Lab Guide
  
  
## Overview:
Deep Learning (DL) is an implementation of Machine Learning (ML) that uses neural networks to solve difficult problems such as image recognition, sentiment analysis and recommendations.  Neural networks simulate the functions of the brain where artificial neurons work in concert to detect patterns in data.  This allows deep learning algorithms to classify, predict and recommend with a high degree of accuracy as more data is trained in the network.  DL algorithms generally operate with a high degree of parallelism and is computationally intense.  As a result, emerging deep learning libraries, frameworks, and platforms allow for data and model parallelization and can leverage advancements in GPU technology for improved performance.  
This workshop will walk you through the deployment of a deep learning library called [MXNet](http://mxnet.io) on AWS using Docker containers.  Containers provide isolation, portability and repeatability, so your developers can easily spin up an environment and start building without the heavy lifting.  

The goal is not to go deep on the learning (no pun intended) aspects, but to illustrate how easy it is to deploy your deep learning environment on AWS and use the same tools to scale your resources as needed.  

### Requirements:  
* AWS account - if you don't have one, it's easy and free to [create one](https://aws.amazon.com/)
* IAM account with elevated privileges allowing you to interact with IAM, EC2, ECS, ECR
* Familiarity with python, Docker, AWS, machine learning - not required but a bonus

### Labs:  
These labs are designed to be completed in sequence.  If you are reading this at a live AWS event, the workshop attendants will give you a high level run down of the labs.  Then it's up to you to follow the instructions below to complete the labs.  Don't worry if you're embarking on this journey in the comfort of your office or home, presentation materials can be found in the git repo.

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
There are just not enough cat picturs on social media these days, to the point where it would be amazing to have a social network dedicated to devoted cat lovers around the world.  Problem is, how do you make sure images uploaded to this niche network are cat related.  Image classification to the rescue!  

Implement MXNet to recognize a variety of images, so you can specifically identify ones of our favorite feline friend!   

Here is the overall architecture of what you will be building throughout this workshop.  By the end of the workshop, you will have the ability to interact directly with the MXNet containers using SSH or Jupyter notebooks.  You will also have the option to create Tasks in ECS which can be run through the management console, CLI or SDKs. 

![Overall Architecture](/images/architecture.png)

### Lab 1 - Set up the Workshop Environment on AWS:    

1\. First you'll need to create an SSH key pair which will be used to login to the instances once provisioned.  Go to the EC2 Dashboard and click on **Key Pairs** in the left menu under Network & Security.  Click **Create Key Pair**, provide a name (can be anything, make it something memorable) when prompted, and click **Create**.  Once created, the private key in the form of .pem file will be automatically downloaded.    

2\. For your convenience, we provide a CloudFormation template to stand up the core infrastructure.  Click on one of these CloudFormation templates to launch your stack:  

Region | Launch Template
------------ | -------------  
**Ohio** (us-east-2) | [![Launch ECS Deep Learning Stack into Ohio with CloudFormation](/images/deploy-to-aws.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-2#/stacks/new?stackName=ecs-deep-learning-stack&templateURL=https://s3.amazonaws.com/BUCKET/TEMPLATE.YAML)  
**Oregon** (us-west-2) | [![Launch ECS Deep Learning Stack into Oregon with CloudFormation](/images/deploy-to-aws.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=ecs-deep-learning-stack&templateURL=https://s3.amazonaws.com/BUCKET/TEMPLATE.YAML)  

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

4\. Generate an SSH key pair.  The container build process will configure public key authentication for SSH access to the container.  The reason we do this is because MXNet has the ability to perform distributed training with multiple machines, and one of the mechanisms used in this case is SSH.  
       
*Note: you will be prompted to enter a passphrase suing the ssh-keygen command, just leave it blank and press enter.*  

<pre>
$ ssh-keygen -t rsa -b 4096 -f id_rsa  
$ cp -av id_rsa* $HOME/.ssh/
</pre>   

5\. Now you're ready to build the Docker image using the provided Dockerfile.

`$ docker build -t mxnet .`  

This process will take about 10-15 minutes because MXNet is being compiled during the build process.  If you're new to Docker, you can take this opportunity to review the Dockerfile to understand what's going on or take a quick break to grab some coffee/tea.  

6\. Once the build process has completed without errors, tag and push the MXNet Docker image to ECR.  You'll reference this image when you deploy the container using ECS in the next lab.  Below is the command and format for the repository URI.  You can find your respository URI in the EC2 Container Service Dashboard; click on **Repositories** in the left menu and click on the repository name that matches the **ecrRepository** output from CloudFormation. The Repository URI will be listed at the top of the screen.  

![ECR URI](/images/ecr-uri.png)  

<pre>
$ docker tag mxnet:latest <b><i>aws_account_id</i></b>.dkr.ecr.<b><i>region</i></b>.amazonaws.com/<b><i>ecr_repository</i></b>:latest   
$ docker push <b><i>aws_account_id</i></b>.dkr.ecr.<b><i>region</i></b>.amazonaws.com/<b><i>ecr_repository</i></b>:latest  
</pre>


**Checkpoint**  
At this point you should have a working MXNet Docker image to deploy with ECS.  If you don't see errors following the commands above, you should be in good shape.  

[Optional] 
An additional test would be to manually run the MXNet image that you just created with an interactive bash session.  

`$ docker run -ti mxnet bash`  

You can exit the session by simply typing **exit**. 


### Lab 3 - Deploy the MXNet Container with ECS:    
Now that you have an MXNet image ready to go, next step is to create a task definition, which specifies parameters used by ECS to run your container, e.g. Docker image, cpu/memory resource requirements, host:container port mappings.  You'll notice that the params in the task definition closely match options passed to a Docker run command.       

1\. Open the EC2 Container Service dashboard, click on **Task Definitions** in the left menu, and click **Create new Task Definition**.    

2\. First, name your task definition, e.g. "mxnet".  If you create a task definition that is a duplicate of an existing task definition, ECS will create a new revision, incrementing the version number automatically.  Next click on **Add container**. 

3\.   



**Checkpoint**  
*add steps to verify successful completion of the lab*  
ssh to master, check connectivity to secondary

### Lab 4 - Image Classification with MXNet:   
Now that you have an MXNet container built and deployed with ECS, you can try out an image classification example provided by MXNet to make sure the framework is working properly.  There are two examples you can run through, one for training a model and one for generating a prediction.  Both examples are presented in the form of a Jupyter notebook.  You may have noticed that Jupyter was installed and configured during the creation of the MXNet image.  If you're new to Jupyter, it is essentially a web application that allows you to interactively step through blocks of written code.  The code can be edited by the user as needed or desired, and there is a play button that lets you step through the cells.  Cells that do not code have no effect, so you can hit play to pass through the cell.          

#### Training:    
*add instructions*

#### Prediction:    
Since training a model can be resource intensive and a lengthy process, you will run through an example that uses a pre-trained model built from the full [ImageNet](http://image-net.org/) dataset, which is a collection of over 10 million images with thousands of classes for those images.  This example is built with a Juypter notebook, so you can interactively walk through the example. 

1\. Open a web browser and visit this URL to access the Jupyter notebook for the demo; the password is ***XXXXXXXXX***:  
http://***ec2_public_dns_name***/notebooks/mxnet-notebooks/python/tutorials/predict_imagenet.ipynb

2\. Play through the cells to run through this example, which loads and prepares the pre-trained model as well as provide methods to load images into the model to predict its classification.  If you've never used Jupyter before, you're probably wonder how you know something is happening.  Cells with code are denoted on the left with "In [n]" where n is simply a cell number.  When you run a cell that takes time to process, the number will show an asterisk.  

**IMPORTANT**: In cell 2, the default context is to use gpu, but in the case of this workshop, we're using cpu resources so change the text "gpu" to "cpu".  This is a great feature of this particular framework, which makes it very easy to switch as needed.  See the following screenshot to see this highlighted; also highlighted in the screenshot is the play button which lets you advance through the cells.  While deep learning performance is better on gpu, you can make use of cpu resources in dev/test environments to keep costs down.  

![Jupyter Notebook - Prediction](/images/jupyter-notebook-predict.png)

3\. Once you've stepped through the two examples at the end of the notebook, try feeding arbitrary images to how well the model performs.  Remember that Jupyter notebooks let you input your own code and run

### Extra Credit Challenges:
At this point, you've run through a couple examples to confirm sample training and prediction tasks are working as expected.  Now it's up to you to use your creativity to take what you've built and expand on it.  

* explore GPUs instances and train using GPUs to see the speed boost
* explore other MXNet algorithms such as matrix factorization

* * *

## Finished!  
Congratulations on completing the lab...*or at least giving it a good go*!  This is the workshop's permananent home, so feel free to revisit as often as you'd like.  In typical Amazon fashion, we'll be listening to your feedback and iterating to make it better.  If you have feedback, we're all ears!  Make sure you clean up after the workshop, so you don't have any unexpected charges on your next monthly bill.  

* * *

## Workshop Cleanup:

1. Delete any manually created resources throughout the labs.
2. Delete any data files stored on S3 and container images stored in ECR.
3. Delete the CloudFormation stack launched at the beginning of the workshop


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



