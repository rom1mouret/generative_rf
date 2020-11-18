package generative_rf

import (
  "github.com/sjwhitworth/golearn/trees"
  "github.com/sjwhitworth/golearn/base"
  "fmt"
)

func GetLeftChild(node *trees.DecisionTreeNode) *trees.DecisionTreeNode {
  next, ok := node.Children["0"]
  if !ok {
    panic("Unable to access the left child")
  }
  return next
}

func GetRightChild(node *trees.DecisionTreeNode) *trees.DecisionTreeNode {
  next, ok := node.Children["1"]
  if !ok {
    panic("Unable to access the right child")
  }
  return next
}

func AssertFloat(attr base.Attribute) *base.FloatAttribute {
  if pFloatAttr, ok := attr.(*base.FloatAttribute); ok {
    return pFloatAttr
  }
  panic("only float attributes are supported")
}

func GetAttributeSpec(attr base.Attribute, data base.FixedDataGrid) base.AttributeSpec {
  spec, err := data.GetAttribute(attr)
  if err != nil {
    panic(fmt.Sprintf("Unable to get attribute's specs: %s", err.Error()))
  }
  return spec
}

func ConcatData(data1 base.FixedDataGrid, data2 base.FixedDataGrid) *base.DenseInstances {
  // build a set of size nAttr x (len(data1) + len(data2))
  data3 := base.NewStructuralCopy(data1)
  _, nRows1 := data1.Size()
  _, nRows2 := data2.Size()
  data3.Extend(nRows1 + nRows2)

  // align specifications
  specs1 := base.ResolveAllAttributes(data1)
  specs2 := base.ResolveAllAttributes(data2) // possibly not aligned with specs1
  specs3 := base.ResolveAllAttributes(data3) // already aligned with specs1

  orderedSpecs2 := make([]base.AttributeSpec, len(specs2))
  for i := 0; i < len(specs3); i++ {
    for k := 0; k < len(specs2); k++ {
      if specs2[k].GetAttribute().Equals(specs3[i].GetAttribute()) {
        orderedSpecs2[i] = specs2[k]
        break
      }
    }
  }

  // add the rows from data1
  data1.MapOverRows(specs1, func(v [][]byte, row int) (bool, error) {
    for i, val := range v {
      data3.Set(specs3[i], row, val)
    }
    return true, nil
  })

  // add the rows from data2
  data2.MapOverRows(orderedSpecs2, func(v [][]byte, row int) (bool, error) {
    row += nRows1
    for i, val := range v {
      data3.Set(specs3[i], row, val)
    }
    return true, nil
  })

  return data3
}

func CopyColumn(fromAttr base.Attribute, from base.FixedDataGrid, to base.UpdatableDataGrid) error {
  var toAttr base.Attribute = nil
  for _, attr := range to.AllClassAttributes() {
      if fromAttr.Equals(attr) {
        toAttr = attr
        break
      }
  }
  if toAttr == nil {
    return fmt.Errorf("cannot find attribute %s in 'to' datagrid", fromAttr.GetName())
  }

  // attribute specs
  fromSpecs, err := from.GetAttribute(fromAttr)
  if err != nil {
    return err
  }
  toSpecs, err := to.GetAttribute(toAttr)
  if err != nil {
    return err
  }

  // copy the values
  _, nRows := from.Size()
  for row := 0; row < nRows; row++ {
    to.Set(toSpecs, row, from.Get(fromSpecs, row))
  }

  return nil
}
