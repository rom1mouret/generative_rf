package main

import (
  "fmt"
  "os"
  "github.com/sjwhitworth/golearn/ensemble"
  "github.com/sjwhitworth/golearn/base"
  "github.com/rom1mouret/generative_rf/golearn/generative_rf"
)

func printAttributes(data base.FixedDataGrid){
  for _, attr := range base.NonClassAttributes(data) {
    fmt.Println("any-kind attribute:", attr.GetName())
  }
  for _, attr := range base.NonClassFloatAttributes(data) {
    fmt.Println("float attribute:", attr.GetName())
  }
  for _, attr := range data.AllClassAttributes() {
    fmt.Println("class attribute:", attr.GetName())
  }
}

func readDataBatches() []base.UpdatableDataGrid {
  // read the training data
  filename := os.Args[1]
  trainData, err := base.ParseCSVToInstances(filename, true)
  if err != nil {
    fmt.Printf("error while reading %s: %s\n", filename, err.Error())
    os.Exit(1)
  }
  specs := base.ResolveAllAttributes(trainData)
  printAttributes(trainData)

  //////////////// divide the data into batches ///////////////
  _, nRows := trainData.Size()
  batchSize := 1024
  nBatches := nRows / batchSize
  fmt.Printf("number of rows: %d; number of batches: %d\n", nRows, nBatches)

  // create batches of size nArgs x batchSize
  batches := make([]base.UpdatableDataGrid, nBatches)
  for j := 0; j < nBatches; j++ {
    batch := base.NewStructuralCopy(trainData)
    batch.Extend(batchSize)
    batches[j] = batch
  }

  // copy the data
  trainData.MapOverRows(specs, func(v [][]byte, row int) (bool, error) {
    j := row / batchSize
    if j < nBatches {  // the last incomplete batch is thrown away
      batch := batches[j]
      batchRow := row % batchSize
      for i, val := range v {
        batch.Set(specs[i], batchRow, val)
      }
    }
    return true, nil
  })

  return batches
}

func writeResults(data base.FixedDataGrid, file *os.File) {
  attr := data.AllClassAttributes()[0].(*base.CategoricalAttribute)
  spec, err := data.GetAttribute(attr)
  if err != nil {
    fmt.Printf("error while retrieving spec to write results: %s\n",  err.Error())
    os.Exit(1)
  }
  _, nRows := data.Size()
  for row := 0; row < nRows; row++ {
    val := attr.GetStringFromSysVal(data.Get(spec, row))
    str := fmt.Sprintf("%s\n", val)
    file.WriteString(str)
  }
}

func main() {
    // read the training data
    batches := readDataBatches()
    dim := len(batches[0].AllAttributes()) - 1

    // prepare the result files
    refFile, err1 := os.Create("ref_results.txt")
    predFile, err2 := os.Create("pred_results.txt")
    if err1 != nil || err2 != nil {
      if err1 != nil {
        fmt.Println(err1.Error())
      } else{
        fmt.Println(err2.Error())
      }
      os.Exit(1)
    }
    defer refFile.Close()
    defer predFile.Close()

    // main loop
    var gen generative_rf.FeatGenerator
    var rf  *ensemble.RandomForest
    for j, batch := range batches {
      classAttr := batch.AllClassAttributes()[0]
      newRF := ensemble.NewRandomForest(50, dim)
      if j == 0 {
        // bootstrap
        rf = newRF
        fmt.Println("fitting the first forest")
        err := rf.Fit(batch)
        if err != nil {
          fmt.Printf("error while training: %s\n", err.Error())
          os.Exit(1)
        }
        fmt.Println("registering the first forest")
        gen.Register(rf, classAttr)
      } else{
        // save predictions for evaluation
        fmt.Printf("batch %d: predicting\n", j+1)
        predictions, err := rf.Predict(batch)
        if err != nil {
          fmt.Printf("error while predicting: %s\n", err.Error())
          os.Exit(1)
        }
        writeResults(predictions, predFile)
        writeResults(batch, refFile)

        // generate data
        fmt.Printf("batch %d: generating features\n", j+1)
        generated, err := gen.Generate(-1, 100)
        if err != nil {
          fmt.Printf("error while generating data: %s\n", err.Error())
          os.Exit(1)
        }

        // fit a new forest
        fmt.Printf("batch %d: fitting forest\n", j+1)
        bigbatch := generative_rf.ConcatData(generated, batch)
        rf = newRF
        err = rf.Fit(bigbatch)
        if err != nil {
          fmt.Printf("error while training on big batch: %s\n", err.Error())
          os.Exit(1)
        }
        fmt.Printf("batch %d: registering forest\n", j+1)
        gen.Register(rf, classAttr).Reinforce(generated)
      }
      gen.Reinforce(batch).UpdateMoments(batch)
    }
}
