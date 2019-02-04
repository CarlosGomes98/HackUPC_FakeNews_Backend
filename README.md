# HackUPC Fake-o-meter Backend

Fake-o-meter is a Google Chrome extension which tells you how reliable the news story you are reading is with a score from 0(completely fake) to 10(completely reliable). To achieve this, we considered 3 main factors: overall style and consistency of the article, credibility of the source of the article and if other sources have discussed the topic of the article.

This is the backend for the extension, that does all the heavy lifting.

Developed at HackUPC, a 36 hour hackathon!

Frontend at: https://github.com/hendrikmolder/fake-o-meter

Check out our devpost: https://devpost.com/software/hackupc_fakenews_backend

![demo](https://raw.githubusercontent.com/CarlosGomes98/HackUPC_FakeNews_Backend/master/demo/image_1.jpg)

![demo](https://raw.githubusercontent.com/CarlosGomes98/HackUPC_FakeNews_Backend/master/demo/image_2.jpg)

### Run it!

Clone this repo and the fake-o-meter repo

Install the requirements in requirements.txt

Run app.py

Follow the instructions at the fake-o-meter repo

### How we built it
In order to judge the consistency of the article, and how well the body describes the title, we trained a machine learning model on a dataset of 20000 fake and credible news articles. For each article, we extract the tf-idf vector of the title, tf-idf vector of the body and the cosine similarity between both as features for a linear regression. Intuitively, this tackles two different aspects.

Tf-idf vectors allow us to judge which words carry the most weight in a document, when put in perspective of the set which contains it.

Firstly, we notice that a lot of "fake" news exist in order to attract clicks. If the title and the body discuss vastly different topics, the article is likely unreliable. Secondly, we observe that "fake" news often share a writing style. By comparing the tf-idf vector of the article with the tf-idf vectors of articles in our dataset, we use that writing style to judge the credibility of the article.

To incorporate the credibility of the source, we built a whitelist of trusted news sources, which get a bonus to their score. Articles from sources with a good reputation have a higher likelihood of being trustworthy.

To compare the article to similar ones from other sources, we use the Google News API. We then extract the tf-idf vectors from them and compare with the original article using cosine similarity. If other news outlets have reported on similar stories, it is more likely to be a real one.