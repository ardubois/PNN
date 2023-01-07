defmodule NNTest do
  use ExUnit.Case
  doctest NN

  test "greets the world" do
    assert NN.hello() == :world
  end
end
