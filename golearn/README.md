The only direct dependency is [GoLearn](https://github.com/sjwhitworth/golearn).

## Simple example

For a simple example, i.e. a single call to the `Generate` function, refer to [single_iteration.go](example/single_iteration.go)

## Continual learning regression on a data stream

It is currently not possible to use generative_rf's Go implementation for regression.

At the time I am writing this, [GoLearn](https://github.com/sjwhitworth/golearn) does not yet support RandomForests with CART regressors.

## Continual learning classification on a data stream

There is no equivalent to `class_sampling` function provided by the Python implementation.
This wouldn't be particularly useful as [GoLearn's RandomForests](https://github.com/sjwhitworth/golearn/blob/master/ensemble/randomforest.go) don't support sample weights for training.

```go
import (
  "github.com/sjwhitworth/golearn/base"
  "github.com/sjwhitworth/golearn/ensemble"
  "github.com/rom1mouret/generative_rf/golearn/generative_rf"
)

func Testing(batches []base.FixedDataGrid) {
  var gen generative_rf.FeatGenerator
  for j, batch := range batches {
    rf := ensemble.NewRandomForest(50, dim)
    classAttr := batch.AllClassAttributes()[0]
    if j > 0 {
      // -1 means nSamples='auto', and at least 500 rows
      generated, _ := gen.Generate(-1, 500)  
      fulldata = generative_rf.ConcatData(generated, batch)
      rf.Fit(fulldata)
      gen.Register(rf, classAttr).Reinforce(generated)
    } else {
      rf.Fit(batch)
      gen.Register(rf, classAttr)
    }
    gen.Reinforce(batch).UpdateMoments(batch)
    // possible ways of using the newly trained rf:
    // 1. compare predictions with the ground truth to detect anomalies
    // 2. serialize it and run it wherever there is no ground truth available
  }
}
```

Below is a more advanced setting where data drift is constantly monitored.
I have also rewritten the for-loop to simulate a realistic streaming scenario.

```go
func Streaming() {
  var gen generative_rf.FeatGenerator
  var driftDetector some_awesome_lib.DriftDetector

  consumer := streaming.StreamConsumer("broker:9092")
  for j, batch := 0, consumer.Poll(); batch != nil; j, batch = j+1, consumer.Poll(){
    if j == 0 || driftDetector.DataDriftDetected(batch) {
      rf := ensemble.NewRandomForest(50, dim)
      classAttr := batch.AllClassAttributes()[0]
      if j > 0 {
        generated, _ := gen.Generate(-1, 500)  
        fulldata = generative_rf.ConcatData(generated, batch)
        rf.Fit(fulldata)
        gen.Register(rf, classAttr).Reinforce(generated)
      } else {
        rf.Fit(batch)
        gen.Register(rf, classAttr)
      }
    }
    gen.Reinforce(batch).UpdateMoments(batch)
  }
}
```
