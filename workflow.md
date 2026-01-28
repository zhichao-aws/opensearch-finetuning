---
config:
  theme: neutral
---
graph TD
    User((User)) -->|Input| CFN[CloudFormation Stack]
    CFN -->|Deploys| SF[Step Functions Workflow]
    SF -->|"1. Start Prep"| DataPrep
    
    subgraph DataPrep ["Data Preparation Phase"]
        direction TB
        subgraph DataDiscovery ["1.1 Data Discovery & Extraction"]
            InputType{Input Type?}
            InputType -->|AOS Index| Extract[Extract from AOS]
            InputType -->|S3 Corpus| S3Check[Validate S3]
            Extract & S3Check --> S3Raw([S3 Raw Corpus])
        end
        DataDiscovery --> DataGen
        subgraph DataGen ["1.2 Synthetic Query Generation"]
            S3Raw --> Bedrock[Bedrock Batch<br/>Inference]
            Bedrock --> PostProc[Generated<br/>Queries JSONL]
            PostProc --> TrainData[Final Training Data<br/>query-doc pairs]
        end
    end
    DataPrep -->|"2. Start Training"| TrainingPhase
    
    subgraph TrainingPhase ["Training Phase (SageMaker)"]
        TrainJob[SageMaker Training Job]
        TrainJob -->|Contrastive Loss| ModelArtifact["Model Artifacts<br/>s3://..."]
    end

    TrainingPhase -->|"3. Start Deployment"| DeployPhase

    subgraph DeployPhase ["Deployment & Registration Phase"]
        Deploy[Deploy SM Endpoint]
        Deploy -->|Real-time| SM_EP[SageMaker Endpoint]
        SM_EP --> Register[Lambda: Register]
        Register -->|ML API| AOS[OpenSearch Service]
        AOS -->|Return| ModelID([Model ID])
    end

    ModelID --> Output[Final Output to User]