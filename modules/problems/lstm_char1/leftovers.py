
#def sample(preds, temperature=1.0):
   ## helper function to sample an index from a probability array
#    preds = np.asarray(preds).astype('float64')
    #preds = np.log(preds) / temperature
    #exp_preds = np.exp(preds)
    #preds = exp_preds / np.sum(exp_preds)
    #probas = np.random.multinomial(1, preds, 1)
    #return np.argmax(probas)

# train the model, output generated text after each iteration
#for iteration in range(1, 60):
    #print()
    #print('-' * 50)
    #print('Iteration', iteration)
    #model.fit(X, y,
              #batch_size=128,
              #epochs=1)

    #start_index = random.randint(0, len(text) - maxlen - 1)

    #for diversity in [0.2, 0.5, 1.0, 1.2]:
        #print()
        #print('----- diversity:', diversity)

        #generated = ''
        #sentence = text[start_index: start_index + maxlen]
        #generated += sentence
        #print('----- Generating with seed: "' + sentence + '"')
        #sys.stdout.write(generated)

        #for i in range(400):
            #x = np.zeros((1, maxlen, len(chars)))
            #for t, char in enumerate(sentence):
                #x[0, t, char_indices[char]] = 1.

            #preds = model.predict(x, verbose=0)[0]
#            next_index = sample(preds, diversity)
            #next_char = indices_char[next_index]

    #        generated += next_char
   #         sentence = sentence[1:] + next_char

  #          sys.stdout.write(next_char)
 #           sys.stdout.flush()
#print()
