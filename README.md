# Large Language Models for Information Extraction: IEPile


# Datasets
\* denotes the dataset is multimodal. # refers to the number of categories or sentences.
<br>

<table>
    <thead>
        <tr>
            <th align="center">Task</th>
            <th align="center">Dataset</th>
            <th align="center">Domain</th>
            <th align="center">#Class</th>
            <th align="center">#Train</th>
            <th align="center">#Val</th>
            <th align="center">#Test</th>
            <th align="center">Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center" rowspan="32" ><strong>NER</strong></td>
            <td align="center">ACE04</td>
            <td align="center">News</td>
            <td align="center">7</td>
            <td align="center">6202</td>
            <td align="center">745</td>
            <td align="center">812</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2005T09">Link</a></td>
        </tr>
        <tr>
            <td align="center">ACE05</td>
            <td align="center">News</td>
            <td align="center">7</td>
            <td align="center">7299</td>
            <td align="center">971</td>
            <td align="center">1060</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2006T06">Link</a></td>
        </tr>
        <tr>
            <td align="center">BC5CDR</td>
            <td align="center">Biomedical</td>
            <td align="center">2</td>
            <td align="center">4560</td>
            <td align="center">4581</td>
            <td align="center">4797</td>
            <td align="center"><a href="https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/">Link</a></td>
        </tr>
        <tr>
            <td align="center">Broad Twitter Corpus</td>
            <td align="center">Social Media</td>
            <td align="center">3</td>
            <td align="center">6338</td>
            <td align="center">1001</td>
            <td align="center">2000</td>
            <td align="center"><a href="https://github.com/GateNLP/broad_twitter_corpus?">Link</a></td>
        </tr>
        <tr>
            <td align="center">CADEC</td>
            <td align="center">Biomedical</td>
            <td align="center">1</td>
            <td align="center">5340</td>
            <td align="center">1097</td>
            <td align="center">1160</td>
            <td align="center"><a href="https://data.csiro.au/collection/csiro:10948?v=3&d=true">Link</a></td>
        </tr>
        <tr>
            <td align="center">CoNLL03</td>
            <td align="center">News</td>
            <td align="center">4</td>
            <td align="center">14041</td>
            <td align="center">3250</td>
            <td align="center">3453</td>
            <td align="center"><a href="https://www.clips.uantwerpen.be/conll2003/ner/">Link</a></td>
        </tr>
        <tr>
            <td align="center">CoNLLpp</td>
            <td align="center">News</td>
            <td align="center">4</td>
            <td align="center">14041</td>
            <td align="center">3250</td>
            <td align="center">3453</td>
            <td align="center"><a href="https://github.com/ZihanWangKi/CrossWeigh">Link</a></td>
        </tr>
        <tr>
            <td align="center">CrossNER-AI</td>
            <td align="center">Artificial Intelligence</td>
            <td align="center">14</td>
            <td align="center">100</td>
            <td align="center">350</td>
            <td align="center">431</td>
            <td align="center" rowspan="5" ><a href="https://github.com/zliucr/CrossNER">Link</a></td>
        </tr>
        <tr>
            <td align="center">CrossNER-Literature</td>
            <td align="center">Literary</td>
            <td align="center">12</td>
            <td align="center">100</td>
            <td align="center">400</td>
            <td align="center">416</td>
        </tr>
        <tr>
            <td align="center">CrossNER-Music</td>
            <td align="center">Musical</td>
            <td align="center">13</td>
            <td align="center">100</td>
            <td align="center">380</td>
            <td align="center">465</td>
        </tr>
        <tr>
            <td align="center">CrossNER-Politics</td>
            <td align="center">Political</td>
            <td align="center">9</td>
            <td align="center">199</td>
            <td align="center">540</td>
            <td align="center">650</td>
        </tr>
        <tr>
            <td align="center">CrossNER-Science</td>
            <td align="center">Scientific</td>
            <td align="center">17</td>
            <td align="center">200</td>
            <td align="center">450</td>
            <td align="center">543</td>
        </tr>
        <tr>
            <td align="center">FabNER</td>
            <td align="center">Scientific</td>
            <td align="center">12</td>
            <td align="center">9435</td>
            <td align="center">2182</td>
            <td align="center">2064</td>
            <td align="center"><a href="https://huggingface.co/datasets/DFKI-SLT/fabner">Link</a></td>
        </tr>
        <tr>
            <td align="center">Few-NERD</td>
            <td align="center">General</td>
            <td align="center">66</td>
            <td align="center">131767</td>
            <td align="center">18824</td>
            <td align="center">37468</td>
            <td align="center"><a href="https://github.com/thunlp/Few-NERD">Link</a></td>
        </tr>
        <tr>
            <td align="center">FindVehicle</td>
            <td align="center">Traffic</td>
            <td align="center">21</td>
            <td align="center">21565</td>
            <td align="center">20777</td>
            <td align="center">20777</td>
            <td align="center"><a href="https://github.com/GuanRunwei/VehicleFinder-CTIM">Link</a></td>
        </tr>
        <tr>
            <td align="center">GENIA</td>
            <td align="center">Biomedical</td>
            <td align="center">5</td>
            <td align="center">15023</td>
            <td align="center">1669</td>
            <td align="center">1854</td>
            <td align="center"><a href="https://github.com/ljynlp/W2NER?tab=readme-ov-file#3-dataset">Link</a></td>
        </tr>
        <tr>
            <td align="center">HarveyNER</td>
            <td align="center">Social Media</td>
            <td align="center">4</td>
            <td align="center">3967</td>
            <td align="center">1301</td>
            <td align="center">1303</td>
            <td align="center"><a href="https://github.com/brickee/HarveyNER">Link</a></td>
        </tr>
        <tr>
            <td align="center">MIT-Movie</td>
            <td align="center">Social Media</td>
            <td align="center">12</td>
            <td align="center">9774</td>
            <td align="center">2442</td>
            <td align="center">2442</td>
            <td align="center"><a href="https://tianchi.aliyun.com/dataset/145106">Link</a></td>
        </tr>
        <tr>
            <td align="center">MIT-Restaurant</td>
            <td align="center">Social Media</td>
            <td align="center">8</td>
            <td align="center">7659</td>
            <td align="center">1520</td>
            <td align="center">1520</td>
            <td align="center"><a href="https://tianchi.aliyun.com/dataset/145105">Link</a></td>
        </tr>
        <tr>
            <td align="center">MultiNERD</td>
            <td align="center">Wikipedia</td>
            <td align="center">16</td>
            <td align="center">134144</td>
            <td align="center">10000</td>
            <td align="center">10000</td>
            <td align="center"><a href="https://github.com/Babelscape/multinerd">Link</a></td>
        </tr>
        <tr>
            <td align="center">NCBI</td>
            <td align="center">Biomedical</td>
            <td align="center">4</td>
            <td align="center">5432</td>
            <td align="center">923</td>
            <td align="center">940</td>
            <td align="center"><a href="https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/">Link</a></td>
        </tr>
        <tr>
            <td align="center">OntoNotes 5.0</td>
            <td align="center">General</td>
            <td align="center">18</td>
            <td align="center">59924</td>
            <td align="center">8528</td>
            <td align="center">8262</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2013T19">Link</a></td>
        </tr>
        <tr>
            <td align="center">ShARe13</td>
            <td align="center">Biomedical</td>
            <td align="center">1</td>
            <td align="center">8508</td>
            <td align="center">12050</td>
            <td align="center">9009</td>
            <td align="center"><a href="https://physionet.org/content/shareclefehealth2013/1.0/">Link</a></td>
        </tr>
        <tr>
            <td align="center">ShARe14</td>
            <td align="center">Biomedical</td>
            <td align="center">1</td>
            <td align="center">17404</td>
            <td align="center">1360</td>
            <td align="center">15850</td>
            <td align="center"><a href="https://physionet.org/content/shareclefehealth2014task2/1.0/">Link</a></td>
        </tr>
        <tr>
            <td align="center">SNAP<sup>*<sup></td>
            <td align="center">Social Media</td>
            <td align="center">4</td>
            <td align="center">4290</td>
            <td align="center">1432</td>
            <td align="center">1459</td>
            <td align="center"><a href="https://www.modelscope.cn/datasets/caijiong_sijun/MoRE-processed-data/files">Link</a></td>
        </tr>
        <tr>
            <td align="center">Temporal Twitter Corpus (TTC)</td>
            <td align="center">Social Meida</td>
            <td align="center">3</td>
            <td align="center">10000</td>
            <td align="center">500</td>
            <td align="center">1500</td>
            <td aligh="center"><a href="https://github.com/shrutirij/temporal-twitter-corpus">Link</a></td>
        </tr>
        <tr>
            <td align="center">Tweebank-NER</td>
            <td align="center">Social Media</td>
            <td align="center">4</td>
            <td align="center">1639</td>
            <td align="center">710</td>
            <td align="center">1201</td>
            <td aligh="center"><a href="https://github.com/mit-ccc/TweebankNLP">Link</a></td>
        </tr>
        <tr>
            <td align="center">Twitter2015<sup>*<sup></td>
            <td align="center">Social Media</td>
            <td align="center">4</td>
            <td align="center">4000</td>
            <td align="center">1000</td>
            <td align="center">3357</td>
            <td align="center"><a href="https://www.modelscope.cn/datasets/caijiong_sijun/MoRE-processed-data/files">Link</a></td>
        </tr>
        <tr>
            <td align="center">Twitter2017<sup>*<sup></td>
            <td align="center">Social Media</td>
            <td align="center">4</td>
            <td align="center">3373</td>
            <td align="center">723</td>
            <td align="center">723</td>
            <td align="center"><a href="https://www.modelscope.cn/datasets/caijiong_sijun/MoRE-processed-data/files">Link</a></td>
        </tr>
        <tr>
            <td align="center">TwitterNER7</td>
            <td align="center">Social Media</td>
            <td align="center">7</td>
            <td align="center">7111</td>
            <td align="center">886</td>
            <td align="center">576</td>
            <td aligh="center"><a href="https://huggingface.co/datasets/tner/tweetner7">Link</a></td>
        </tr>
        <tr>
            <td align="center">WikiDiverse<sup>*<sup></td>
            <td align="center">News</td>
            <td align="center">13</td>
            <td align="center">6312</td>
            <td align="center">755</td>
            <td align="center">757</td>
            <td aligh="center"><a href="https://github.com/wangxw5/wikidiverse?tab=readme-ov-file#get-the-data">Link</a></td>
        </tr>
        <tr>
            <td align="center">WNUT2017</td>
            <td align="center">Social Media</td>
            <td align="center">6</td>
            <td align="center">3394</td>
            <td align="center">1009</td>
            <td align="center">1287</td>
            <td aligh="center"><a href="https://tianchi.aliyun.com/dataset/144349">Link</a></td>
        </tr>
        <tr>
            <td align="center" rowspan="11"><strong>RE</strong></td>
            <td align="center">ACE05</td>
            <td align="center">News</td>
            <td align="center">7</td>
            <td align="center">10051</td>
            <td align="center">2420</td>
            <td align="center">2050</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2006T06">Link</a></td>
        </tr>
        <tr>
            <td align="center">ADE</td>
            <td align="center">Biomedical</td>
            <td align="center">1</td>
            <td align="center">3417</td>
            <td align="center">427</td>
            <td align="center">428</td>
            <td align="center"><a href="http://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/">Link</a></td>
        </tr>
        <tr>
            <td align="center">CoNLL04</td>
            <td align="center">News</td>
            <td align="center">5</td>
            <td align="center">922</td>
            <td align="center">231</td>
            <td align="center">288</td>
            <td align="center"><a href="http://lavis.cs.hs-rm.de/storage/spert/public/datasets/conll04/">Link</a></td>
        </tr>
        <tr>
            <td align="center">DocRED</td>
            <td align="center">Wikipedia</td>
            <td align="center">96</td>
            <td align="center">3008</td>
            <td align="center">300</td>
            <td align="center">700</td>
            <td align="center"><a href="https://github.com/thunlp/DocRED">Link</a></td>
        </tr>
        <tr>
            <td align="center">MNRE<sup>*<sup></td>
            <td align="center">Social Media</td>
            <td align="center">23</td>
            <td align="center">12247</td>
            <td align="center">1624</td>
            <td align="center">1614</td>
            <td align="center"><a href="https://github.com/thecharm/MNRE">Link</a></td>
        </tr>
        <tr>
            <td align="center">NYT</td>
            <td align="center">News</td>
            <td align="center">24</td>
            <td align="center">56196</td>
            <td align="center">5000</td>
            <td align="center">5000</td>
            <td align="center"><a href="https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_nyt10.sh">Link</a></td>
        </tr>
        <tr>
            <td align="center">Re-TACRED</td>
            <td align="center">News</td>
            <td align="center">40</td>
            <td align="center">58465</td>
            <td align="center">19584</td>
            <td align="center">13418</td>
            <td align="center"><a href="https://github.com/gstoica27/Re-TACRED">Link</a></td>
        </tr>
        <tr>
            <td align="center">SciERC</td>
            <td align="center">Scientific</td>
            <td align="center">7</td>
            <td align="center">1366</td>
            <td align="center">187</td>
            <td align="center">397</td>
            <td align="center"><a href="http://lavis.cs.hs-rm.de/storage/spert/public/datasets/scierc/">Link</a></td>
        </tr>
        <tr>
            <td align="center">SemEval2010</td>
            <td align="center">General</td>
            <td align="center">19</td>
            <td align="center">6507</td>
            <td align="center">1493</td>
            <td align="center">2717</td>
            <td align="center"><a href="https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_semeval.sh">Link</a></td>
        </tr>
        <tr>
            <td align="center">TACRED</td>
            <td align="center">News</td>
            <td align="center">42</td>
            <td align="center">68124</td>
            <td align="center">22631</td>
            <td align="center">15509</td>
            <td align="center"><a href="https://nlp.stanford.edu/projects/tacred/">Link</a></td>
        </tr>
        <tr>
            <td align="center">TACREV</td>
            <td align="center">News</td>
            <td align="center">42</td>
            <td align="center">68124</td>
            <td align="center">22631</td>
            <td align="center">15509</td>
            <td align="center"><a href="https://github.com/DFKI-NLP/tacrev">Link</a></td>
        </tr>
        <tr>
            <td align="center" rowspan="7"><strong>EE</strong></td>
            <td align="center">ACE05</td>
            <td align="center">News</td>
            <td align="center">33/22</td>
            <td align="center">17172</td>
            <td align="center">923</td>
            <td align="center">832</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2006T06">Link</a></td>
        </tr>
        <tr>
            <td align="center">CASIE</td>
            <td align="center">Cybersecurity</td>
            <td align="center">5/26</td>
            <td align="center">11189</td>
            <td align="center">1778</td>
            <td align="center">3208</td>
            <td align="center"><a href="https://github.com/Ebiquity/CASIE">Link</a></td>
        </tr>
        <tr>
            <td align="center">GENIA11</td>
            <td align="center">Biomedical</td>
            <td align="center">9/11</td>
            <td align="center">8730</td>
            <td align="center">1091</td>
            <td align="center">1092</td>
            <td align="center"><a href="https://bionlp-st.dbcls.jp/GE/2011/eval-test/">Link</a></td>
        </tr>
        <tr>
            <td align="center">GENIA13</td>
            <td align="center">Biomedical</td>
            <td align="center">13/7</td>
            <td align="center">4000</td>
            <td align="center">500</td>
            <td align="center">500</td>
            <td align="center"><a href="https://bionlp-st.dbcls.jp/GE/2013/eval-test/">Link</a></td>
        </tr>
        <tr>
            <td align="center">PHEE</td>
            <td align="center">Biomedical</td>
            <td align="center">2/16</td>
            <td align="center">2898</td>
            <td align="center">961</td>
            <td align="center">968</td>
            <td align="center"><a href="https://github.com/ZhaoyueSun/PHEE">Link</a></td>
        </tr>
        <tr>
            <td align="center">RAMS</td>
            <td align="center">News</td>
            <td align="center">139/65</td>
            <td align="center">7329</td>
            <td align="center">924</td>
            <td align="center">871</td>
            <td align="center"><a href="https://nlp.jhu.edu/rams/">Link</a></td>
        </tr>
        <tr>
            <td align="center">WikiEvents</td>
            <td align="center">Wikipedia</td>
            <td align="center">50/59</td>
            <td align="center">5262</td>
            <td align="center">378</td>
            <td align="center">492</td>
            <td align="center"><a href="https://github.com/raspberryice/gen-arg">Link</a></td>
        </tr>
    </tbody>
</table>