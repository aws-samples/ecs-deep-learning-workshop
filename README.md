# Deploy a Deep Learning Framework on Amazon ECS: Lab Guide
  
  
## Overview
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
**Lab 2:** Build an MXNet Container  
**Lab 3:** Deploy the MXNet Container with ECS  
**Lab 4:** Image Classification with MXNet  

### Workshop Cleanup
This section will appear again below as a reminder because you will be deploying infrastructure on AWS which will have an associated cost.  Fortunately, this workshop should take no more than 2 hours to complete, so costs are minimal.  See the appendix for an estimate of what this workshop should cost to run.    

1. Delete any manually created resources throughout the labs.
2. Delete any data files stored on S3 and container images stored in ECR.
3. Delete the CloudFormation stack launched at the beginning of the workshop
  
* * * 

## Let's Begin!  

### Your Challenge  
There are just not enough cat picturs on social media these days, to the point where it would be amazing to have a social network dedicated to devoted cat lovers around the world.  Problem is, how do you make sure images uploaded to this niche network are cat related.  Image classification to the rescue!  

Implement MXNet to recognize a variety of images, so you can specifically identify ones of our favorite feline friend!   

Here is the overall architecture of what you will be building throughout this workshop.  By the end of the workshop, your developers will have the ability to interact directly with the MXNet containers using SSH or Jupyter notebooks.  They will also have the option to create Tasks in ECS which can be run through the management console, CLI or SDKs. 

![Workshop Architecture](/images/architecture.png)

### Lab 1 - Set up the Workshop Environment on AWS    

1\. First you'll need to create an SSH key pair which will be used to login to the instances once provisioned.  Go to the EC2 Dashboard and click on **Key Pairs** in the left menu under Network & Security.  Click **Create Key Pair**, provide a name (can be anything, make it something memorable) when prompted, and click **Create**.  Once created, the private key in the form of .pem file will be automatically downloaded.    

2\. For your convenience, we provide a CloudFormation template to stand up the core infrastructure.  Click on one of these CloudFormation templates to launch your stack:  

Region | Launch Template
------------ | -------------  
**Ohio** (us-east-2) | [![Launch ECS Deep Learning Stack into Ohio with CloudFormation](/images/deploy-to-aws.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-2#/stacks/new?stackName=ecs-deep-learning-stack&templateURL=https://s3.amazonaws.com/BUCKET/TEMPLATE.YAML)  
**Oregon** (us-west-2) | [![Launch ECS Deep Learning Stack into Oregon with CloudFormation](/images/deploy-to-aws.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=ecs-deep-learning-stack&templateURL=https://s3.amazonaws.com/BUCKET/TEMPLATE.YAML)  

The template will automatically bring you to the CloudFormation Dashboard and start the stack creation process in the specified region.  The template sets up a VPC, IAM roles, S3 bucket, ECR container registry, and an ECS cluster which is comprised of two EC2 instances with the Docker daemon running on each.  The idea is to provide a contained environment, so as not to interfere with any other things in your account.   

**Checkpoint**  
In the CloudFormation Dashboard, your stack should show status CREATE\_COMPLETE.  

*INSERT SCREENSHOT HERE*

If there was an error during the stack creation process, CloudFormation will rollback and terminate.  You can investigate and troubleshoot by looking for errors in the Events tab.     

### Lab 2 - Build an MXNet Container  
In this lab, you will build an MXNet docker container using one of the ECS cluster instances which already comes bundled with Docker installed.  There are quite a few dependencies for MXNet, so for your convenience, we provide a Dockerfile in the lab 2 folder to make sure nothing is missed.  MXNet uses SSH as the mechanism for communication between containers, so you'll be generating an SSH key pair to configure public key authentication for secure access.  You can review the Dockerfile to see what's being installed.  Links to MXNet documentation can be found in the Appendix if you'd like to read more about it.  

1\. You will build the container in one of the EC2 instances from the ECS cluster.  Go to the EC2 Dashboard in the Management Console.  Select one of the EC2 instances created by the CloudFormation stack, and note the Public DNS name. 

*INSERT SCREENSHOT HERE*

SSH into the host  
`ssh -i <private-key.pem> ec2-user@<ec2-public-DNS-name>`

2\. Once logged into the EC2 instance, clone the workshop github repository so you can easily access the Dockerfile.  
`git clone https://github.com/awslabs/ecs-deep-learning-workshop.git`  

3\. Navigate to the lab-2-build/mxnet/ folder to use as our working directory.  
`cd ecs-deep-learning-workshop/lab-2-build/mxnet`

4\. Generate an SSH key pair.  The container build process will configure public key authentication for SSH access to the container.  Later in the workshop, you'll see that training commands will be issued over SSH.       
`ssh-keygen -t rsa -b 4096 -f id_rsa
cp -av id_rsa* $HOME/.ssh/`  

5\. Now you're ready to build the Docker image based on the provided Dockerfile.    
`docker build -t mxnet .`

6\. Once the build process has completed, manually run the container to make sure everything looks good.  
`docker run -ti mxnet bash`

7\. Push the MXNet image to ECR.  You'll use this to deploy the container using ECS in the next lab.  
``   



**Checkpoint**  
*add steps to verify successful completion of the lab*  
run the container in interactive mode with bash shell 
test SSH access to the instance

### Lab 3 - Deploy the MXNet Container with ECS  
Now that you have an MXNet container ready to go, you will create a task definition to deploy your shiny new container with ECS.  Task definitions specify key parameters used by ECS to run your container.  For example, it specifies which Docker image to deploy, cpu/memory resource requirements, port mappings, volume mappings and more.  Task Definitions can be represented in JSON, so for your convenience, we provide a task definition in the lab 3 folder for you to use.       

*add instructions...*

**Checkpoint**  
*add steps to verify successful completion of the lab*  
ssh to master, check connectivity to secondary

### Lab 4 - Image Classification with MXNet 
Now you can start playing with MXNet.  There are two main sections in this lab - training the model and clasifying images.  We must first train the model in order for the neural network to learn from a large collection of sample images.  *add detail about data set*.  

Since training can be a lengthy process, we provide a pre-trained model for you to test image classification.  You will use a Jupyter notebook to interactive work with the model to generate classifications of provided images.  

*add instructions*

**Checkpoint**  
*add steps to verify successful completion of the lab*  


### Extra Credit Challenges:
* explore GPUs instances and train using GPUs to see the speed boost
* explore other MXNet algorithms such as matrix factorization

* * *

## Finished!  
Congratulations on completing the lab...*or at least giving it a good go*!  This is the workshop's permananent home, so feel free to revisit as often as you'd like.  In typical Amazon fashion, we'll be listening to your feedback and iterating to make it better.  If you have feedback, we're all ears!  Make sure you clean up after the workshop, so you don't have any unexpected charges on your next monthly bill.  

* * *

## Workshop Cleanup

1. Delete any manually created resources throughout the labs.
2. Delete any data files stored on S3 and container images stored in ECR.
3. Delete the CloudFormation stack launched at the beginning of the workshop


* * *

## Appendix

### Cost  
Here are estimated costs for running this 2 hour workshop:   
*add cost estimate for EC2 and other*  

### AWS Resources
Check out these links to learn more about the services used in this workshop  
- [AWS Services](https://aws.amazon.com/)  
- [A Cloud Guru self-paced AWS labs](https://acloud.guru/courses)  

https://aws.amazon.com/blogs/compute/powering-your-amazon-ecs-clusters-with-spot-fleet/


### Docker Resources
If you're wondering if a container is the same as a VM, you're on the right track, but the truth shall set you free.  Have a look here to learn more about containerization technology and why it's all the rage.  
- [Docker](https://www.docker.com/)  
*add links to ECS sessions*  

### Machine Learning / Deep Learning / MXNet Resources 
If you're wondering how MXNet was able to classify all the Internet cat pics you threw at it, you can learn more about the library and the concepts here.  
- [MXNet](http://mxnet.io/)  
- [MXNet Examples](http://mxnet.io/tutorials/index.html)  
*add link to Werner's email about MXNet and Amazon*
*add link to MAC401 session*  
*add links to Coursera classes on ML and neuralnetworks*  


https://aws.amazon.com/blogs/compute/distributed-deep-learning-made-easy/


