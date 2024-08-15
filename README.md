# Readme file
MODEL INPUT/OUTPUTS

## Training Pipeline
SENT TO PROCESSING POD:

```
{
    "payload": 100
}
```

DATA I WANT FROM PROCESSING POD:

```
{
    "success": True
}
```

DATA I SEND TO INFERENCE POD:

```
{
    "payload": 100
}
```

DATA I WANT FROM INFERENCE POD:

```
{
    "success": True,
    "accuracy": 0.95,  # Example model accuracy
    "parameters": {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32
    }
}
```

## Inference Pipeline
SENT TO INFERENCE POD:

```
{
    "images": [
        {
            "image": "/9j/4AAQSkZJRgABAQEAAAAAAAD/4QCSRXhpZgAATU0AKgAAAAgAA...",
            "filename": "image1.png"
        },
        {
            "image": "/9j/4AAQSkZJRgABAQEAAAAAAAD/4QCSRXhpZgAATU0AKgAAAAgAA...",
            "filename": "image2.jpg"
        }
    ],
    "payload": "200"
}
```

WHAT I WANT FROM INFERENCE POD:

```
[
    {
        "imageUrl": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA...",
        "prediction": "Cat",
        "confidence": 0.95
    },
    {
        "imageUrl": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA...",
        "prediction": "Dog",
        "confidence": 0.89
    }
]
```


## Port designations:

Flask:
> 5000 --> Container
> FLOATING-IP --> Nodeport 

Inference:
> 83 --> Exposed Port
> 8083 --> targetPort

DP:
> 84 --> Exposed Port
> 8084 --> targetPort

MT:
> 85 --> Exposed Port
> 8085 --> targetPort