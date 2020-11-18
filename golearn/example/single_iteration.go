package main

import (
  "fmt"
  "os"
  "github.com/sjwhitworth/golearn/ensemble"
  "github.com/sjwhitworth/golearn/base"
  "github.com/rom1mouret/generative_rf/golearn/generative_rf"
)


func main() {
    // read the dataset (without headers)
    dataset, _ := base.ParseCSVToInstances("blobs.csv", false)
    dim, _ := dataset.Size()
    dim -= 1

    // fit a golearn's random forest on the dataset
    rf := ensemble.NewRandomForest(100, dim)
    err := rf.Fit(dataset)
    if err != nil {
       panic(err)
    }

    // generate data
    var gen generative_rf.FeatGenerator
    gen.Register(rf, dataset.AllClassAttributes()[0]).Reinforce(dataset).UpdateMoments(dataset)
    data, _ := gen.Generate(1000, -1)

    // write the generated data
    file, _ := os.Create("generated.csv")
    defer file.Close()

    specs := base.ResolveAllAttributes(data)
    data.MapOverRows(specs, func(v [][]byte, row int) (bool, error) {
      for col := 0; col < dim; col++ {
          floatv := base.UnpackBytesToFloat(v[col])
          file.WriteString(fmt.Sprintf("%f,", floatv))
      }
      val := specs[dim].GetAttribute().GetStringFromSysVal(v[dim])
      file.WriteString(fmt.Sprintf("%s\n", val))
      return true, nil
    })
}
