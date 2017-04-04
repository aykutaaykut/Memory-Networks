using Knet, AutoGrad, Compat, ArgParse

function parse_commandline()
  s = ArgParseSettings()
  @add_arg_table s begin
    ("--datafiles"; nargs = '+'; help = "If provided, use first file for training, second for dev, others for test.")
    ("--input"; arg_type = Char; default = 's'; help = "s for sentence sequences, w for word sequences as input")
    ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
    ("--seed"; arg_type=Int; default=38; help="Random number seed.")
    ("--atype"; default = (gpu() >= 0 ? "KnetArray{Float32}" : "Array{Float32}"); help = "Array type: Array for CPU, KnetArray for GPU")
  end
  return parse_args(s; as_symbols = true)
end

function main(args = ARGS)
  #Fixed configurations for baseline model
  embedding_dimension = 100
  learning_rate = 0.01
  margin = 0.1
  total_epochs = 10

  #User settings are added.
  settings = parse_commandline()
  println("settings", [(symbol, value) for (symbol, value) in settings]...)
  settings[:seed] > 0 && srand(settings[:seed])
  settings[:atype] = eval(parse(settings[:atype]))

  #Classification of datafiles as training, dev and test.
  training_data = settings[:datafiles][1]
  if length(settings[:datafiles]) > 1
    dev_data = settings[:datafiles][2]
  end
  if length(settings[:datafiles]) > 2
    test_data = settings[:datafiles][3:end]
  end

  #Creating the dictionary of words in the data.
  dict = createDict(training_data)
  dict_length = length(dict)
  info("$dict_length unique words.")
  feature_space = 3 * dict_length

  model = initWeights(settings[:atype], feature_space, embedding_dimension, settings[:winit])
  mappingMatrix = settings[:winit] * randn(feature_space, feature_space)

  for epoch = 1:total_epochs
    @time uo, ur, avg_loss = train(training_data, model[:uo], model[:ur], feature_space, dict,
                                   mappingMatrix, learning_rate, margin, settings[:atype])
    model[:uo] = uo
    model[:ur] = ur
    println("(epoch: $epoch, loss: $avg_loss)")
  end
end

function createDict(training_data)
  dict = Dict{String, Int}()
  number = 1
  f = open(training_data)
  while !eof(f)
    str = readline(f)
    number, dict = parseLineAddDict(number, str, dict)
  end
  close(f)
  return dict
end

function parseLineAddDict(number, line, dict)
  words = split(line)
  for i = 1:length(words)
    str = words[i]
    if str[end] == '?' || str[end] == '.'
      str = str[1:end - 1]
    end
    if !haskey(dict, str)
      dict[str] = number
      number = number + 1
    end
  end
  return number, dict
end

function initWeights(atype, feature_space, embedding_dimension, winit)
  weights = Dict{Symbol, Any}()

  uo = winit * randn(embedding_dimension, feature_space)
  ur = winit * randn(embedding_dimension, feature_space)
  weights[:uo] = uo
  weights[:ur] = ur

  for k in keys(weights)
    weights[k] = convert(atype, weights[k])
  end
  return weights
end

function I(x)
  return x
end

function G(x, memory)
  feature_rep = I(x)
  push!(memory, feature_rep)
  return memory
end

function O(x, memory, uo, d, dict, mappingMatrix, atype)
  scorelist1 = so(x, uo, d, memory, dict, mappingMatrix, atype)
  o1 = indmax(scorelist1)
  mo1 = memory[o1]
  input2 = [x, mo1]
  scorelist2 = so(input2, uo, d, memory, dict, mappingMatrix, atype)
  o2 = indmax(scorelist2)
  mo2 = memory[o2]
  return [x, mo1, mo2]
end

function R(input, dict, mappingMatrix, ur, d, atype)
  reverseDict = Array(String, length(dict))
  for (key, value) in dict
    reverseDict[value] = key
  end

  scorelist = sr(input, ur, d, dict, mappingMatrix, atype)
  answer = indmax(scorelist)
  return reverseDict[answer]
end



#mode = 1 for phix and x comes from the input
#mode = 2 for phix and x comes from a supporting memory
#mode = 3 for phiy
function onehot(x, dict, d, mode)
  words = split(x)
  onehots = zeros(d, length(words))
  if mode == 1
    for i = 1:length(words)
      word = words[i]
      if word[end] == '?' || word[end] == '.'
        word = word[1:end - 1]
      end
      value = dict[word]
      onehots[value, i] = 1
    end
  else
    if mode == 2
      for i = 1:length(words)
        word = words[i]
        if word[end] == '?' || word[end] == '.'
          word = word[1:end - 1]
        end
        value = dict[word]
        onehots[length(dict) + value, i] = 1
      end
    else
      for i = 1:length(words)
        word = words[i]
        if word[end] == '?' || word[end] == '.'
          word = word[1:end - 1]
        end
        value = dict[word]
        onehots[(2 *length(dict)) + value, i] = 1
      end
    end
  end
  return onehots
end

#mode = 1 for phix and x comes from the input
#mode = 2 for phix and x comes from a supporting memory
#mode = 3 for phiy
function phi(atype, x, d, dict, mappingMatrix, mode)
  onehots = onehot(x, dict, d, mode)
  mapped = zeros(d, 1)
  for i = 1:size(onehots, 2)
    onehot = onehots[:, i]
    mapped = mapped + mappingMatrix * onehot
  end
  mapped = convert(atype, mapped)
  return mapped
end

function s(x, y, u, d, dict, mappingMatrix, atype)
  score = 0
  phiy = phi(atype, y, d, dict, mappingMatrix, 3)
  if typeof(x) == String
    phix = phi(atype, x, d, dict, mappingMatrix, 1)
    current_score = phix' * u' * u * phiy
    current_score = sum(current_score)
    score = score + current_score
  else
    for i = 1:length(x)
      input = x[i]
      if i == 1
        phix = phi(atype, input, d, dict, mappingMatrix, 1)
      else
        phix = phi(atype, input, d, dict, mappingMatrix, 2)
      end
      current_score = phix' * u' * u * phiy
      current_score = sum(current_score)
      score = score + current_score
    end
  end
  return score
end

function so(x, uo, d, memory, dict, mappingMatrix, atype)
  scorelist = Any[]
  for i = 1:length(memory)
    score = s(x, memory[i] , uo, d, dict, mappingMatrix, atype)
    push!(scorelist, score)
  end
  return scorelist
end

function sr(x, ur, d, dict, mappingMatrix, atype)
  scorelist = Any[]
  for k in keys(dict)
    score = s(x, k, ur, d, dict, mappingMatrix, atype)
    push!(scorelist, score)
  end
  return scorelist
end

function answer(x, memory, uo, ur, d, dict, mappingMatrix, atype)
  output = O(x, memory, uo, d, dict, mappingMatrix, atype)
  answer = R(output, dict, mappingMatrix, ur, d, atype)
  return answer
end

function resetMemory()
  return Any[]
end

function marginRankingLoss(comb, memory, x, gold_labels, d, dict, mappingMatrix, margin, atype)
  uo = comb[1]
  ur = comb[2]
  total_loss = 0
  m1_loss = 0
  m2_loss = 0
  r_loss = 0

  correct_m1 = gold_labels[1]
  correct_m2 = gold_labels[2]
  correct_r = gold_labels[3]

  for i = 1:length(memory)
    if memory[i] != correct_m1
      m1l = max(0, margin - s(x, correct_m1, uo, d, dict, mappingMatrix, atype) + s(x, memory[i], uo, d, dict, mappingMatrix, atype))
      m1_loss = m1_loss + m1l
    end
  end

  input_m = [x, correct_m1]
  for j = 1:length(memory)
    if memory[j] != correct_m2
      m2l = max(0, margin - s(input_m, correct_m2, uo, d, dict, mappingMatrix, atype) + s(input_m, memory[j], uo, d, dict, mappingMatrix, atype))
      m2_loss = m2_loss + m2l
    end
  end

  input_r = [x, correct_m1, correct_m2]
  for k in keys(dict)
    if k != correct_r
      rl = max(0, margin - s(input_r, correct_r, ur, d, dict, mappingMatrix, atype) + s(input_r, k, ur, d, dict, mappingMatrix, atype))
      r_loss = r_loss + rl
    end
  end

  total_loss = (m1_loss + m2_loss + r_loss)
  return total_loss
end

marginRankingLossGradient = grad(marginRankingLoss)

function train(data_file, uo, ur, d, dict, mappingMatrix, lr, margin, atype)
  total_loss = 0
  numq = 0
  memory = resetMemory()
  f = open(data_file)
  while !eof(f)
    str = readline(f)
    words = split(str)
    if words[end][end] == '.'
      line_number = words[1]
      sentence = words[2]
      for i = 3:length(words)
        sentence = sentence * " " * words[i]
      end
      if line_number == 1
        memory = resetMemory()
      end
      memory = G(sentence, memory)
    else
      line_number = words[1]
      line_number = parse(Int, line_number)
      question = words[2]
      for i = 3:(length(words) - 3)
        question = question * " " * words[i]
      end
      memory = G(question, memory)

      correct_r = words[length(words) - 2]

      correct_m1_index = words[length(words) - 1]
      correct_m1_index = parse(Int, correct_m1_index)
      correct_m1 = memory[length(memory) - (line_number - correct_m1_index)]

      correct_m2_index = words[length(words)]
      correct_m2_index = parse(Int, correct_m2_index)
      correct_m2 = memory[length(memory) - (line_number - correct_m2_index)]

      gold_labels = [correct_m1, correct_m2, correct_r]
      comb = []
      push!(comb, uo)
      push!(comb, ur)
      loss = marginRankingLoss(comb, memory, question, gold_labels, d, dict, mappingMatrix, margin, atype)
      lossGradient = marginRankingLossGradient(comb, memory, question, gold_labels, d, dict, mappingMatrix, margin, atype)

      uo = copy!(uo, uo - lr * lossGradient[1])
      ur = copy!(ur, ur - lr * lossGradient[2])
      
      total_loss = total_loss + loss
      numq = numq + 1
    end
  end
  close(f)
  avg_loss = total_loss / numq
  return uo, ur, avg_loss
end

main()
