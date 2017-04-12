using Knet, AutoGrad, Compat, ArgParse

function parse_commandline()
  s = ArgParseSettings()
  @add_arg_table s begin
    ("--datafiles"; nargs = '+'; help = "If provided, use first file for training, second for test.")
    ("--winit"; arg_type=Float64; default=0.01; help="Initial weights set to winit*randn().")
    ("--seed"; arg_type=Int; default=38; help="Random number seed.")
    ("--atype"; default = (gpu() >= 0 ? "KnetArray{Float64}" : "Array{Float64}");
                                        help = "Array type: Array for CPU, KnetArray for GPU")
  end
  return parse_args(s; as_symbols = true)
end

function main(args = ARGS)
  #Fixed configurations for baseline model
  embedding_dimension = 100
  learning_rate = 0.01
  margin = 0.1
  total_epochs = 10 #Actual value is 10

  #User settings are added.
  settings = parse_commandline()
  println("settings", [(symbol, value) for (symbol, value) in settings]...)
  settings[:seed] > 0 && srand(settings[:seed])
  settings[:atype] = eval(parse(settings[:atype]))

  #Classification of datafiles as training and test.
  training_data = settings[:datafiles][1]
  if length(settings[:datafiles]) > 1
    test_data = settings[:datafiles][2]
  end

  #Creating the dictionary of words in the data.
  vocabDict = createDict([training_data, test_data])
  vocabDict_length = length(vocabDict)
  info("$vocabDict_length unique words.")
  feature_space = 3 * vocabDict_length

  model = initWeights(settings[:atype], feature_space, embedding_dimension, settings[:winit])

  for epoch = 1:total_epochs
    @time avg_loss = train(training_data, model[:uo], model[:ur], vocabDict,
                                     learning_rate, margin, settings[:atype])
    @time accuracy = test(test_data, model[:uo], model[:ur], vocabDict, settings[:atype])
    println("[Training => (epoch: $epoch, loss: $avg_loss)] , [Test => (epoch: $epoch, accuracy: $accuracy %)]")
  end
end

function createDict(datafiles)
  dict = Dict{String, Int}()
  number = 1
  for file in datafiles
    f = open(file)
    while !eof(f)
      str = readline(f)
      number, dict = parseLineAddDict(number, str, dict)
    end
    close(f)
  end
  return dict
end

function parseLineAddDict(number, line, dict)
  words = split(line)
  if words[end][end] == '.'
    for i = 2:length(words)
      str = words[i]
      if str[end] == '.'
        str = str[1:end - 1]
      end
      if !haskey(dict, str)
        dict[str] = number
        number = number + 1
      end
    end
  else
    for i = 2:length(words) - 1
      str = words[i]
      if str[end] == '?'
        str = str[1:end - 1]
      end
      if !haskey(dict, str)
        dict[str] = number
        number = number + 1
      end
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

function I(x, dict, atype)
  words = split(x)
  feature_rep = zeros(Float64, length(dict), 1)
  for w in words
    if w[end] == '?' || w[end] == '.'
      w = w[1:end - 1]
    end
    onehot = word2OneHot(w, dict)
    feature_rep = feature_rep .+ onehot
  end
  feature_rep = convert(atype, feature_rep)
  return feature_rep
end

function word2OneHot(word, dict)
  onehot = zeros(Float64, length(dict), 1)
  for w in keys(dict)
    if w == word
      onehot[dict[w], 1] = 1.0
      break
    end
  end
  return onehot
end

function G(feature_rep, memory)
  push!(memory, feature_rep)
  return memory
end

function O(x_feature_rep, memory, uo, atype)
  x_feature_rep_list = [x_feature_rep]
  scoreDict1 = so(x_feature_rep_list, memory, uo, atype)
  o1 = scoreDict1[maximum(keys(scoreDict1))]
  mo1 = memory[o1]

  return [x_feature_rep, mo1]
end

#mode = 1 for phix and x comes from the input
#mode = 2 for phix and x comes from a supporting memory
#mode = 3 for phiy
function phi(feature_rep, mode, atype)
  mapped = zeros(Float64, 3 * length(feature_rep), 1)
  if mode == 1
    for i = 1:length(feature_rep)
      mapped[i] = feature_rep[i]
    end
  else
    if mode == 2
      for i = 1:length(feature_rep)
        mapped[length(feature_rep) + i] = feature_rep[i]
      end
    else
      for i = 1:length(feature_rep)
        mapped[2 * length(feature_rep) + i] = feature_rep[i]
      end
    end
  end
  mapped = convert(atype, mapped)
  return mapped
end

function s(x_feature_rep_list, y_feature_rep, u, atype)
  score = 0
  phiy = phi(y_feature_rep, 3, atype)
  if length(x_feature_rep_list) == 1
    phix = phi(x_feature_rep_list[1], 1, atype)
    current_score = phix' * u' * u * phiy
    current_score = sum(current_score)
    score = score + current_score
  else
    for i = 1:length(x_feature_rep_list)
      input = x_feature_rep_list[i]
      if i == 1
        phix = phi(input, 1, atype)
      else
        phix = phi(input, 2, atype)
      end
      current_score = phix' * u' * u * phiy
      current_score = sum(current_score)
      score = score + current_score
    end
  end
  return score
end

function so(x_feature_rep_list, memory, uo, atype)
  scoreDict = Dict{Float64, Int}()
  for i = 1:length(memory)
    score = s(x_feature_rep_list, memory[i], uo, atype)
    scoreDict[score] = i
  end
  return scoreDict
end

function R(input_list, vocabDict, ur, atype)
  scoreDict = sr(input_list, vocabDict, ur, atype)
  answer = scoreDict[maximum(keys(scoreDict))]
  return answer
end

function sr(x_feature_rep_list, vocabDict, ur, atype)
  scoreDict = Dict{Float64, String}()
  for k in keys(vocabDict)
    y_feature_rep = word2OneHot(k, vocabDict)
    score = s(x_feature_rep_list, y_feature_rep, ur, atype)
    scoreDict[score] = k
  end
  return scoreDict
end

function marginRankingLoss(comb, x_feature_rep, memory, vocabDict, gold_labels, margin, atype)
  uo = comb[1]
  ur = comb[2]
  total_loss = 0
  m1_loss = 0
  m2_loss = 0
  r_loss = 0

  correct_m1 = gold_labels[1]
  correct_r = gold_labels[2]

  input_1 = [x_feature_rep]
  for i = 1:length(memory)
    if memory[i] != correct_m1
      m1l = max(0, margin - s(input_1, correct_m1, uo, atype) + s(input_1, memory[i], uo, atype))
      m1_loss = m1_loss + m1l
    end
  end

  correct_r_feature_rep = word2OneHot(correct_r, vocabDict)
  input_r = [x_feature_rep, correct_m1]
  for k in keys(vocabDict)
    if k != correct_r
      k_feature_rep = word2OneHot(k, vocabDict)
      rl = max(0, margin - s(input_r, correct_r_feature_rep, ur, atype) + s(input_r, k_feature_rep, ur, atype))
      r_loss = r_loss + rl
    end
  end

  total_loss = m1_loss + r_loss
  return total_loss
end

marginRankingLossGradient = grad(marginRankingLoss)

function train(data_file, uo, ur, vocabDict, lr, margin, atype)
  total_loss = 0
  numq = 0
  memory = resetMemory()
  f = open(data_file)
  while !eof(f)
    str = readline(f)
    words = split(str)
    if words[end][end] == '.'
      line_number = words[1]
      line_number = parse(Int, line_number)
      sentence = words[2]
      for i = 3:length(words)
        if words[i][end] == '?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        sentence = sentence * " " * words[i]
      end
      if line_number == 1
        memory = resetMemory()
      end
      sentence_feature_rep = I(sentence, vocabDict, atype)
      G(sentence_feature_rep, memory)
    else
      line_number = words[1]
      line_number = parse(Int, line_number)
      question = words[2]
      for i = 3:(length(words) - 2)
        if words[i][end] == '?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        question = question * " " * words[i]
      end
      question_feature_rep = I(question, vocabDict, atype)
      G(question_feature_rep, memory)

      correct_r = words[end - 1]

      correct_m1_index = words[end]
      correct_m1_index = parse(Int, correct_m1_index)
      correct_m1 = memory[correct_m1_index]

      gold_labels = [correct_m1, correct_r]
      comb = [uo, ur]
      loss = marginRankingLoss(comb, question_feature_rep, memory, vocabDict, gold_labels, margin, atype)
      lossGradient = marginRankingLossGradient(comb, question_feature_rep, memory, vocabDict, gold_labels, margin, atype)

      copy!(uo, uo - lr * lossGradient[1])
      copy!(ur, ur - lr * lossGradient[2])

      total_loss = total_loss + loss
      numq = numq + 1
    end
  end
  close(f)
  avg_loss = total_loss / numq
  return avg_loss
end

function resetMemory()
  return Any[]
end

function test(data_file, uo, ur, vocabDict, atype)
  numcorr = 0
  numq = 0
  memory = resetMemory()
  f = open(data_file)
  while !eof(f)
    str = readline(f)
    words = split(str)
    if words[end][end] == '.'
      line_number = words[1]
      line_number = parse(Int, line_number)
      sentence = words[2]
      for i = 3:length(words)
        if words[i][end] == '?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        sentence = sentence * " " * words[i]
      end
      if line_number == 1
        memory = resetMemory()
      end
      sentence_feature_rep = I(sentence, vocabDict, atype)
      G(sentence_feature_rep, memory)
    else
      line_number = words[1]
      line_number = parse(Int, line_number)
      question = words[2]
      for i = 3:(length(words) - 2)
        if words[i][end] == '?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        question = question * " " * words[i]
      end
      question_feature_rep = I(question, vocabDict, atype)
      G(question_feature_rep, memory)

      correct_r = words[end - 1]

      response = answer(question_feature_rep, memory, vocabDict, uo, ur, atype)
      if response == correct_r
        numcorr = numcorr + 1
      end

      numq = numq + 1
    end
  end
  close(f)
  accuracy = numcorr / numq * 100
  return accuracy
end

function answer(x_feature_rep, memory, vocabDict, uo, ur, atype)
  output = O(x_feature_rep, memory, uo, atype)
  answer = R(output, vocabDict, ur, atype)
  return answer
end

main()
