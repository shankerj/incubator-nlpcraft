<img src="https://nlpcraft.org/images/nlpcraft_logo_black.gif" height="80px">
<br>
<a target=_ href="https://gitter.im/apache-nlpcraft/community"><img alt="Gitter" src="https://badges.gitter.im/apache-nlpcraft/community.svg"></a>

### Docker
Starting with version 0.7.3 NLPCraft provides docker image for NLPCraft server. You can 
also prepare your own images following the instructions below. 

#### Using Server Image
 1. [Install](https://docs.docker.com/install/) docker.
 2. Pull actual NLPCraft server image: ```docker pull nlpcraftserver/server:0.7.3```, 
 where `0.7.3` is the desired version.
 3. Run image: ```docker run -m 8G -p 8081:8081 -p 8201:8201 -p 8202:8202 nlpcraftserver/server:0.7.3```

**NOTE**:
 Default docker image run REST server with default configuration (see file `build/nlpcraft.conf`). See [configuration](https://nlpcraft.org/server-and-probe.html)
 documentation on how to override default configuration using environment variables. 

#### Your Own Server Image
You can prepare your own NLPCraft server image following these instructions:
 1. Build the release with maven: ```mvn clean package -P release```  
 2. Prepare docker image with `prepare.sh` (modify it if necessary):
   ```./prepare.sh docker_acct server 0.7.3```, where
    * `0.7.3` - maven build version, prepared on previous step.
    * `docker_acct` - change it to your own [DockerHub](https://hub.docker.com) account.    
 
    Note that docker image name will be `docker_acct/server:0.7.3`   
 3. Login to [DockerHub](https://hub.docker.com): ```docker login```   
 4. Push prepared docker image: ```docker push docker_acct/server:0.7.3```  
 5. Logout from [DockerHub](https://hub.docker.com): ```docker logout```

**NOTE**:
 1. Under MacOS - increase docker limit [resources configuration](https://docs.docker.com/config/containers/resource_constraints/). NLPCraft server requires more memory
than allocated by default to the docker containers. Set memory value to 8G (`-m 8G`). 
          
### Copyright
Copyright (C) 2020 Apache Software Foundation

<img src="https://www.apache.org/img/ASF20thAnniversary.jpg" height="64px">

