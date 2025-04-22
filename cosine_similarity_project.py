"""
Standart optimizasyon koşulları altında gerçekleşmiş kosinüs benzerliği algoritması ile TF-IDF işlemi ile meti vektöre dönüştürme
işlemini optimize ediniz. Optimize edilmiş algoritma ile metinlerin benzerliğini hesaplayınız. Sonuçları karşılaştırınız.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stopWords = [
        "acaba", "ama", "ancak", "artık", "aslında", "az", "bazı", "belki", "ben", "biri",
        "birkaç", "birşey", "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye",
        "eğer", "en", "gibi", "hem", "hep", "hepsi", "her", "hiç", "için", "ile", "ise",
        "kez", "ki", "kim", "mı", "mu", "mü", "nasıl", "ne", "neden", "nerde", "nerede",
        "nereye", "niçin", "niye", "o", "sanki", "şey", "siz", "şu", "tüm", "ve", "veya",
        "ya", "yani"
    ]

def stopWord(kelime):
    return kelime in stopWords

# Örnek cümleler
cumle_1 = "Sanma şahım herkesi sen sadıkane yar olur"
cumle_2 = "Herkesi sen dost mu sandın belki ol ağyar olur"

# TF-IDF ile vektörleştirme ve benzerlik hesaplama
corpus = [cumle_1, cumle_2]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
similarity_tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Manuel stopword kontrolü


def sozlukOku(dizi):
    sozluk = []
    for cumle in dizi:
        kelimeler = cumle.split(" ")
        for kelime in kelimeler:
            if stopWord(kelime):
                continue
            if kelime not in sozluk:
                sozluk.append(kelime)
    return sozluk

# Cümleyi vektöre çevirme
def cumle2vec(cumle, sozluk):
    vector = []
    kelimeler = cumle.split(" ")
    for sozcuk in sozluk:
        sozcukSayi = 0
        for kelime in kelimeler:
            if kelime == sozcuk:
                sozcukSayi += 1
        vector.append([sozcuk, sozcukSayi])
    return vector

# Vektör boyutu hesaplama
def vectorBoyut(vector):
    toplam = 0
    for i in range(len(vector)):
        toplam += vector[i][1] * vector[i][1]
    return toplam ** 0.5

# Noktasal çarpım
def noktasalCarpim(vector1, vector2):
    if len(vector1) != len(vector2):
        return -1
    toplam = 0
    for i in range(len(vector1)):
        toplam += vector1[i][1] * vector2[i][1]
    return toplam

# Kosinüs benzerliği hesaplama
def cosinusBenzerligi(vector1, vector2):
    return noktasalCarpim(vector1, vector2) / (vectorBoyut(vector1) * vectorBoyut(vector2))

# Uygulama
sozluk = sozlukOku([cumle_1, cumle_2])
cumle_1_vector = cumle2vec(cumle_1, sozluk)
cumle_2_vector = cumle2vec(cumle_2, sozluk)
benzerlikOrani = cosinusBenzerligi(cumle_1_vector, cumle_2_vector)

# Sonuçlar
print(f"Kelime frekansına göre kosinüs benzerliği: {benzerlikOrani}")
print(f"TF-IDF ile kosinüs benzerliği: {similarity_tfidf[0][0]}")
