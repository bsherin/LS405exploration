import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def display_matrix(mat, row_labels, col_labels, rows=10):
    return pd.DataFrame(mat, index=row_labels, columns=col_labels).head(rows).round(3)

def matrix_heatmap(mtx, row_labels, col_labels, cmap='YlOrBr'):
    fig=plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
    x_tick_marks = np.arange(len(col_labels))
    y_tick_marks = np.arange(len(row_labels))
    plt.xticks(x_tick_marks, col_labels, fontsize=8, rotation=90)
    plt.yticks(y_tick_marks, row_labels, fontsize=8)
    plt.tick_params("x", top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.imshow(mtx, norm=matplotlib.colors.LogNorm(), interpolation='nearest', cmap=cmap)


from IPython.display import display_html

def display_side_by_side(dfs, round_to=3):
    html_str = ''
    for i, df in enumerate(dfs):
        df_html = df.round(round_to).to_html(index=False)
        html_str += f'<div style="display:inline-block; vertical-align:top; margin-right:10px"><h4>Topic {i}</h4>{df_html}</div>'
    display_html(html_str, raw=True)

def wprob(w, cfdist, total_words):
    return cfdist[w] / total_words

def relevance(w, ld, pwt, cfdist, total_words, use_log=True):
    import math
    if use_log and pwt == 0:
        return -99999
    if use_log:
        return ld * math.log(pwt) + (1 - ld) * math.log(pwt / wprob(w, cfdist, total_words))
    else:
        return ld * pwt + (1 - ld) * (pwt / wprob(w, cfdist, total_words))
    
def display_topics(model, vectorizer, fdist, lbda, n=10, use_log=True):
    total_words = sum(fdist.values())
    feature_names = vectorizer.get_feature_names_out()
    topic_rel_dfs = []
    for topic in model.components_:
        topic_sum = sum(topic)
        word_rel_dict = {}
        for idx, word in enumerate(feature_names):
            pwt = topic[idx] / topic_sum
            word_rel_dict[word] = relevance(word, lbda, pwt, fdist, total_words, use_log=use_log)

        df = pd.DataFrame(sorted(list(word_rel_dict.items()), key=lambda x: x[1], reverse=True)[:n], columns=["word", "relevance"])
        topic_rel_dfs.append(df)
    display_side_by_side(topic_rel_dfs)
    return topic_rel_dfs