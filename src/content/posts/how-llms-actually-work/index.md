---
title: "LLM thực sự hoạt động như thế nào"
description: "Hướng dẫn từng bước, nhẹ nhàng về mặt toán học, giải thích cách các LLM hiện đại dựa trên transformer hoạt động — từ tokenization đến dự đoán token tiếp theo."
published: 2026-07-25
category: "AI"
tags: ["llm", "transformer", "machine learning", "deep learning", "ai"]
draft: false
annotation: ""
---

> **Quy ước thuật ngữ:** Các thuật ngữ technical quan trọng được giữ bằng tiếng Anh, kèm nghĩa tiếng Việt trong ngoặc ở lần xuất hiện đầu tiên.

Bài viết này là phần hướng dẫn từng bước về cách **LLM — Large Language Model (mô hình ngôn ngữ lớn)** hoạt động. Các LLM hiện đại chủ yếu được xây dựng bằng cách xếp chồng nhiều **transformer block (khối transformer)** lên nhau. Vì vậy, hiểu được cơ chế của **transformer** sẽ giúp bạn nắm được phần lớn cách LLM vận hành.

Bài viết sẽ trình bày những cơ chế cốt lõi bên trong các LLM hiện đại dựa trên **transformer**, nhưng không đi sâu vào phần toán học phức tạp. Điều đó không có nghĩa là bạn không nên học toán; tuy nhiên, nội dung này có thể được dùng như một phần nhập môn.

Phần lớn LLM hiện đại có chung một bộ khung thuộc **transformer family (họ kiến trúc transformer)**. Sự khác biệt nằm ở dữ liệu mà từng mô hình được huấn luyện, quy mô, các lựa chọn cấu hình và quá trình **post-training (hậu huấn luyện)** được thực hiện sau đó. Sau khi đọc xong, bạn sẽ có thể xem nhiều bài báo về LLM hoặc **model card (tài liệu mô tả mô hình)** hiện đại và nhận biết mỗi phần đang nói về thành phần nào trong kiến trúc.

## Lộ trình nội dung

1. **Tokens (các đơn vị token):** cách một chuỗi văn bản trở thành một dãy số nguyên.
2. **Embeddings (biểu diễn nhúng):** cách những số nguyên đó mang ý nghĩa.
3. **Positional encoding (mã hóa vị trí):** cách mô hình biết thứ tự xuất hiện của các token.
4. **Attention (cơ chế chú ý):** cách các token trao đổi thông tin với nhau.
5. **Multi-head attention (cơ chế chú ý đa đầu):** cách mô hình theo dõi nhiều loại quan hệ cùng lúc.
6. **Feed-forward network (mạng truyền thẳng):** nơi lưu giữ một phần lớn cấu trúc đã học của mô hình.
7. **Residual stream (luồng phần dư)** và **layer normalization (chuẩn hóa lớp):** những cơ chế giúp các mô hình rất sâu có thể huấn luyện được.
8. **Next-token prediction (dự đoán token tiếp theo):** đầu ra thực sự của mô hình và cách vòng lặp sinh văn bản hoạt động.
9. **Architecture (kiến trúc)** so với **trained weights (trọng số đã huấn luyện):** phần nào được dùng chung rộng rãi giữa các LLM hiện đại và phần nào tạo nên khác biệt.
10. Toàn bộ **transformer pipeline (quy trình transformer)** từ **tokenization (phân tách token)** đến dự đoán token tiếp theo.

Trong suốt bài viết sẽ có các phần giải thích ngắn để người đọc có thể theo dõi bất kể nền tảng kỹ thuật trước đó.

---

![Toàn bộ transformer pipeline](transformer-pipeline.png "Tổng quan toàn bộ transformer pipeline — các phần bên dưới sẽ đi qua từng thành phần một.")

## 1. Tokenization (phân tách token)

Mô hình không trực tiếp đọc văn bản. Chúng đọc các **integer ID (mã số nguyên)**. Bước chuyển câu lệnh của bạn thành một dãy số nguyên được gọi là **tokenization (phân tách token)**.

Một **tokenizer (bộ phân tách token)** nhận vào một chuỗi ký tự và tạo ra một dãy số nguyên. Mỗi số nguyên trỏ tới một mục trong **vocabulary (bộ từ vựng)** cố định. Vocabulary của các LLM hiện đại thường chứa từ vài chục nghìn đến vài trăm nghìn mục.

> **Giải thích ngắn — token ID (mã token)**  
> Token ID là số nguyên mà mô hình sử dụng để đại diện cho một mục trong vocabulary. Mô hình làm việc với con số này, không phải trực tiếp với từ được viết ra.

Token thường không phải là một từ hoàn chỉnh. Chúng thường là các **subword pieces (mảnh từ con)**. Chẳng hạn, từ `tokenization` có thể được tách thành `["token", "ization"]`, còn từ `running` có thể được tách thành `["run", "ning"]`.

Lý do là hiệu quả. Vocabulary dựa hoàn toàn trên từ nguyên vẹn sẽ quá lớn và khó khái quát hóa cho các từ mới. Vocabulary ở cấp ký tự lại quá nhỏ, khiến mô hình phải tự học từ đầu ngay cả những mẫu rất đơn giản. **Subword tokenization (phân tách token theo từ con)** nằm ở giữa hai cách này: các mảnh phổ biến nhất trở thành token riêng, còn các từ hiếm hoặc mới được ghép từ những mảnh nhỏ hơn.

> **Giải thích ngắn — vocabulary (bộ từ vựng)**  
> Vocabulary là danh sách cố định gồm các mảnh mà tokenizer có thể nhận biết. Mỗi mảnh có một ID và mô hình chỉ có thể trực tiếp nhận các ID thuộc danh sách đó.

Sự đánh đổi này xuất hiện ở những chỗ mà nhiều người không ngờ tới. Ví dụ kinh điển là hỏi một LLM có bao nhiêu chữ `r` trong từ `strawberry`. Trước đây, LLM thường trả lời sai. Đây không hẳn là thất bại trong việc đếm, mà là do mô hình không trực tiếp thao tác trên từng chữ cái. Nó chỉ xử lý các token ID, trong khi con người có thể tự nhiên tách từ đó thành từng ký tự.

![Sơ đồ tokenization](transformer-tokenization.png "Tokenization biến văn bản thành token ID.")

Các họ mô hình khác nhau sử dụng những tokenizer khác nhau. Các mô hình GPT sử dụng các biến thể của **Byte Pair Encoding — BPE (mã hóa theo cặp byte)**. **SentencePiece** phổ biến trong các mô hình theo phong cách LLaMA. Lựa chọn này ảnh hưởng đến **compute (khối lượng tính toán)** — ít token hơn đồng nghĩa với ít công việc hơn — cũng như khả năng bao phủ đa ngôn ngữ. Tuy nhiên, hình thức cơ bản vẫn giống nhau: văn bản đi vào, số nguyên đi ra.

Khi prompt đã trở thành một dãy số nguyên, bước tiếp theo là cung cấp ý nghĩa cho các số nguyên đó.

---

## 2. Embeddings (biểu diễn nhúng)

Một token ID như `1024` chỉ là chỉ số của một hàng. Bản thân nó không mang ý nghĩa. Thành phần cung cấp ý nghĩa cho nó là một bảng rất lớn gọi là **embedding matrix (ma trận nhúng)**.

Mỗi mô hình đều có một embedding matrix. Ma trận này có một hàng cho mỗi mục trong vocabulary, và mỗi hàng là một **vector (véc-tơ)** dài gồm nhiều con số. Độ dài của mỗi hàng được gọi là **hidden size (kích thước ẩn)** của mô hình. Trong nhiều mô hình thuộc lớp 7B, mỗi token có thể được biểu diễn bằng 4.096 số. Các mô hình lớn hơn thường sử dụng vector rộng hơn.

> **Giải thích ngắn — vector (véc-tơ)**  
> Vector là một danh sách các con số. Trong transformer, mỗi token trở thành một vector để mô hình có thể thực hiện phép toán trên nó.

Khi tokenizer chuyển một số nguyên cho mô hình, mô hình tra cứu hàng tương ứng trong embedding matrix và sử dụng vector ở hàng đó. Vector này là **embedding (biểu diễn nhúng)** của token — cách mô hình biểu diễn “ý nghĩa” của token, được học trong quá trình huấn luyện.

> **Giải thích ngắn — embedding matrix (ma trận nhúng)**  
> Embedding matrix là một bảng tra cứu: đầu vào là token ID, đầu ra là vector đã được học.

Một đặc tính đáng chú ý là các token có ý nghĩa gần nhau thường có vector gần nhau trong không gian. Vector của `king` nằm gần vector của `queen`, còn vector của `Paris` nằm gần `France`. Không có quan hệ nào trong số này được lập trình cứng. Chúng xuất hiện từ quá trình huấn luyện trên lượng văn bản đủ lớn, bởi mô hình học được rằng những vị trí như vậy giúp nó dự đoán văn bản tốt hơn.

Ta thậm chí có thể thực hiện số học trên embedding và đôi khi nhận được kết quả có ý nghĩa. Ví dụ nổi tiếng:

```text
king − man + woman ≈ queen
```

Hình học của **embedding space (không gian nhúng)** chứa cấu trúc ngữ nghĩa thực sự, mặc dù không ai trực tiếp yêu cầu mô hình phải xây dựng không gian theo cách đó.

![Không gian embedding](transformer-embedding-analogy.png "Minh họa không gian embedding với các quan hệ ngữ nghĩa.")

Cần làm rõ rằng ở giai đoạn này, mỗi token đã được thay bằng embedding của nó, nhưng embedding riêng lẻ không cho biết token nằm ở đâu trong chuỗi. Vector của `dog` vẫn giống nhau dù `dog` là từ đầu tiên hay từ thứ năm trong prompt. Đây là một vấn đề.

**Positional encoding (mã hóa vị trí)** được dùng để lấp khoảng trống này.

---

## 3. Positional encoding (mã hóa vị trí)

**Self-attention (cơ chế tự chú ý)** thuần túy không có sẵn biểu diễn về thứ tự từ. Nếu không có tín hiệu vị trí, mô hình không có cách trực tiếp để biết `dog` xuất hiện trước `bites` hay sau nó.

Thứ tự từ có thể thay đổi ý nghĩa. Vì vậy, mô hình cần một cơ chế đưa vị trí của mỗi token vào phép tính.

> **Giải thích ngắn — positional encoding (mã hóa vị trí)**  
> Positional encoding cung cấp thông tin về thứ tự, cho mô hình biết mỗi token nằm ở đâu trong chuỗi.

Bài báo transformer gốc của Vaswani và cộng sự năm 2017 thực hiện điều này bằng cách gán cho mỗi vị trí một mẫu số riêng, rồi cộng trực tiếp mẫu đó vào embedding của token trước các bước xử lý khác. Vị trí 1 có một mẫu, vị trí 5 có mẫu khác và vị trí 100 lại có một mẫu khác. Các mẫu được tạo từ sóng **sine (sin)** và **cosine (cos)** ở nhiều tần số.

Khi đó, embedding của `dog` ở vị trí 1 khác embedding của `dog` ở vị trí 5, chỉ vì một mẫu vị trí khác đã được cộng vào nó.

Cách này hoạt động tốt. **Sinusoidal encoding (mã hóa hình sin)** được lựa chọn một phần vì có thể **extrapolate (ngoại suy)** ra ngoài độ dài chuỗi chính xác đã xuất hiện trong quá trình huấn luyện. Tuy nhiên, các phương pháp cộng trực tiếp vị trí vẫn có hai vấn đề ngày càng quan trọng khi mô hình được mở rộng.

Thứ nhất, embedding phải đồng thời chứa cả ý nghĩa và vị trí trong cùng một tập hợp số. Khả năng chứa thông tin là hữu hạn.

Thứ hai, **learned absolute position embedding (embedding vị trí tuyệt đối được học)** không khái quát hóa một cách tự nhiên. Nếu mô hình chỉ được huấn luyện với prompt dài tối đa 2.048 token, nó chưa từng thấy vị trí 5.000 trong quá trình huấn luyện, nên embedding cho vị trí đó không được học theo cùng một cách.

Các mô hình hiện đại phần lớn sử dụng một phương pháp khác gọi là **Rotary Position Embeddings — RoPE (embedding vị trí xoay)**, được Su và cộng sự giới thiệu năm 2021 và hiện được sử dụng trong LLaMA, Mistral, Gemma, Qwen cùng nhiều họ mô hình **open-weight (công khai trọng số)** khác.

Trực giác của RoPE như sau: thay vì cộng thông tin vị trí vào vector của token, RoPE xoay các vector **Query (truy vấn)** và **Key (khóa)** theo một góc phụ thuộc vào vị trí token. Token ở vị trí 1 được xoay một góc nhỏ; token ở vị trí 100 được xoay nhiều hơn. Khi hai token được so sánh trong attention, điều quan trọng là độ chênh giữa phép xoay Query và Key, qua đó biểu diễn khoảng cách tương đối giữa chúng.

> **Giải thích ngắn — RoPE**  
> RoPE là viết tắt của Rotary Position Embeddings. Thay vì cộng thêm một vector vị trí, nó xoay Query và Key để khoảng cách tương đối xuất hiện trong quá trình attention.

![Rotary position embeddings](transformer-rope.png "Rotary position embeddings xoay vector theo vị trí.")

Những lợi ích thực tế của RoPE gồm:

- Biểu diễn **relative position (vị trí tương đối)** một cách tự nhiên, phù hợp hơn với nhu cầu của attention.
- Khái quát hóa tốt hơn sang **long context (ngữ cảnh dài)**.
- Không bổ sung thêm **parameter (tham số)** mới cho mô hình.

Dù có positional encoding tốt, các LLM hiện đại vẫn gặp vấn đề được ghi nhận là **lost in the middle (mất thông tin ở giữa)**, theo Liu và cộng sự năm 2023. Mô hình thường sử dụng thông tin ở đầu và cuối một prompt dài đáng tin cậy hơn thông tin bị chôn ở giữa.

Đó là lý do các mẹo **prompt engineering (kỹ thuật thiết kế prompt)** như “đặt ngữ cảnh quan trọng lên đầu” hoặc “nhắc lại thông tin then chốt ở cuối” có thể thực sự hữu ích. Mô hình không sử dụng mọi phần trong prompt với mức độ hiệu quả ngang nhau.

Khi cả ý nghĩa token và vị trí đã được mã hóa, câu hỏi tiếp theo là: các token trao đổi thông tin với nhau như thế nào?

---

## 4. Attention (cơ chế chú ý)

Đây là cơ chế đã tạo nên tên gọi của kiến trúc: **attention (cơ chế chú ý)**.

Bên trong mỗi **transformer layer (lớp transformer)**, attention thực hiện một nhiệm vụ: cho phép mỗi token nhìn vào những token khác mà nó được phép thấy, rồi quyết định token nào quan trọng đối với bước tiếp theo.

Attention thực hiện việc này bằng cách gán cho mỗi token đồng thời ba vai trò. Mỗi token được biến đổi thành ba vector mới: **Query (truy vấn)**, **Key (khóa)** và **Value (giá trị)**, thường viết tắt là **Q, K, V**.

> **Giải thích ngắn — Q, K, V**  
> Query có nghĩa là “tôi đang tìm kiếm điều gì”; Key là “tôi có thể khớp với điều gì”; còn Value là thông tin được sao chép khi mức độ khớp đủ mạnh.

- **Query:** “Tôi đang tìm kiếm thông tin gì từ các token khác?”
- **Key:** “Đây là đặc điểm tôi cung cấp cho những token đang tìm kiếm tôi.”
- **Value:** “Đây là thông tin sẽ được truyền đi khi có sự khớp.”

Cùng một token đồng thời đóng cả ba vai trò. Các phép biến đổi Q, K, V là những **learned matrix (ma trận được học)**, vì vậy trong quá trình huấn luyện, mô hình tự tìm ra mỗi token nên tìm kiếm điều gì và nên cung cấp điều gì.

Việc ghép cặp được thực hiện thông qua một **similarity score (điểm tương đồng)**. Query của mỗi token được so sánh với Key của từng token mà nó được phép thấy bằng **scaled dot product (tích vô hướng có tỷ lệ)**. Theo trực giác, phép tính này đo mức độ hai vector cùng hướng. Hệ số tỷ lệ giúp các con số ổn định trước khi đi qua softmax.

> **Giải thích ngắn — dot product (tích vô hướng)**  
> Dot product là một cách đơn giản để chấm điểm mức độ thẳng hàng của hai vector. Càng thẳng hàng, mức độ khớp càng mạnh.

Các điểm khớp sau đó được chuyển thành trọng số bằng **softmax**. Softmax nhận một tập hợp số bất kỳ và biến chúng thành một phân phối giống xác suất có tổng bằng 1. Token có điểm khớp cao hơn nhận trọng số lớn hơn. Các trọng số này được dùng để tính **weighted average (trung bình có trọng số)** của những Value vector.

> **Giải thích ngắn — softmax**  
> Softmax biến điểm thô thành các trọng số có tổng bằng 1. Điểm lớn nhận trọng số lớn; điểm nhỏ nhận trọng số nhỏ.

Xét câu:

> “The cat that I saw yesterday was sleeping.”

Khi mô hình xử lý từ `was`, nó cần xác định đối tượng nào đang ngủ. Query vector của `was` được so sánh với Key vector của những token nó được phép thấy.

Dot product với `cat` có giá trị cao, vì mô hình đã học rằng những động từ như `was` cần một chủ ngữ và những chủ ngữ như `cat` tạo ra Key vector phù hợp. Dot product với `yesterday` có giá trị thấp. Softmax chuyển các điểm này thành trọng số: `cat` nhận trọng số cao, còn `yesterday` nhận trọng số thấp.

Sau đó, mô hình lấy tổng có trọng số của các Value vector tương ứng. Vì vậy, Value của `cat` chi phối kết quả. Biểu diễn mới của `was` lúc này chịu ảnh hưởng chủ yếu từ thông tin của `cat`. Đây là cách một token ở cách xa nhiều vị trí có thể trở thành đối tượng được quy chiếu.

Có một ràng buộc đặc thù đối với mô hình ngôn ngữ kiểu GPT: chúng sinh văn bản từ trái sang phải. Token ở vị trí 5 chỉ được attention tới các vị trí từ 1 đến 5. Nó không thể nhìn vào vị trí 6, 7 hoặc 8 vì các token đó chưa được sinh ra.

Cơ chế này được gọi là **causal masking (mặt nạ nhân quả)**. Cách triển khai khá đơn giản: các token trong tương lai nhận điểm khớp cực thấp, khiến trọng số của chúng sau softmax gần như bằng 0.

> **Giải thích ngắn — causal masking (mặt nạ nhân quả)**  
> Causal masking che các token trong tương lai, ngăn mô hình ngôn ngữ **decoder-only (chỉ có bộ giải mã)** nhìn trước khi dự đoán token tiếp theo.

![Attention heatmap](transformer-attention-heatmap.png "Attention heatmap (bản đồ nhiệt attention) minh họa causal masking và mức chú ý cao dành cho từ cat.")

Một trong những phát hiện đáng chú ý nhất trong **interpretability research (nghiên cứu khả năng diễn giải)** liên quan đến các attention head chuyên biệt được gọi là **induction head (đầu quy nạp)**, được Anthropic phát hiện năm 2022.

Các head này học cách nhận ra mẫu có dạng `A B … A` trong prompt và dự đoán rằng `B` sẽ xuất hiện tiếp theo. Khi mô hình gặp `A` lần thứ hai, induction head nhìn lại nơi `A` từng xuất hiện, xem điều gì đến sau nó rồi sao chép mẫu đó.

Đây là một trong những cơ chế rõ ràng nhất đã biết đứng sau **in-context learning (học trong ngữ cảnh)** — khả năng của LLM nhận ra một mẫu ngay trong prompt và tiếp tục theo mẫu đó.

> **Giải thích ngắn — induction head (đầu quy nạp)**  
> Induction head là một attention head nhận ra các mẫu lặp trong prompt và giúp tiếp tục chúng.

Attention có một chi phí lớn. Trong **full attention (attention đầy đủ)**, mỗi token so sánh với tất cả token mà nó được phép thấy. Vì vậy, khi độ dài prompt tăng gấp đôi, lượng công việc xấp xỉ tăng gấp bốn lần.

Đây là lý do prompt dài tốn kém khi chạy, đồng thời giải thích vì sao nhiều nghiên cứu gần đây tập trung vào attention hiệu quả hơn, chẳng hạn:

- **FlashAttention**
- **Sparse attention (attention thưa)**
- **Linear attention (attention tuyến tính)**

Tuy nhiên, một attention head chỉ cung cấp cho mô hình một góc nhìn đã học về các mối quan hệ đó.

---

## 5. Multi-head attention (cơ chế chú ý đa đầu)

Một lần chạy attention đơn lẻ chỉ cung cấp một cách để mô hình quyết định token nào quan trọng với token nào. Điều đó là chưa đủ, bởi ngôn ngữ chứa nhiều mối quan hệ diễn ra đồng thời:

- Sự hòa hợp giữa chủ ngữ và động từ.
- Quan hệ giữa đại từ và tên mà nó đề cập.
- Tham chiếu xa giữa các câu.
- Thứ tự từ và các cụm từ cục bộ.

**Multi-head attention (cơ chế chú ý đa đầu)** giải quyết vấn đề này bằng cách chạy nhiều phép attention song song. Mỗi phép chạy song song hoạt động trong một không gian nhỏ riêng và được gọi là một **head (đầu chú ý)**.

> **Giải thích ngắn — attention head (đầu chú ý)**  
> Một attention head là một phép attention độc lập với các phép chiếu đã học riêng của nó.

Có một điểm thường bị mô tả sai, kể cả trong nhiều bài hướng dẫn: mỗi head không đơn giản nhận một lát cắt cố định từ vector token gốc.

Thay vào đó, mỗi head có những **projection matrix (ma trận chiếu)** riêng được học, dùng để ánh xạ toàn bộ vector token xuống các vector Q, K và V nhỏ hơn của chính head đó.

Ví dụ, nếu mô hình có 4.096 số cho mỗi token và 32 head, mỗi head thường làm việc trong không gian 128 chiều. Tuy nhiên, 128 số đó là một phép chiếu được học từ toàn bộ 4.096 số, chứ không phải một đoạn cố định. Nói cách khác, chúng là những “góc nhìn” khác nhau về cùng một token, không phải những mảnh tách biệt của token đó.

Mỗi head chạy attention độc lập. Sau đó, đầu ra của tất cả head được **concatenate (nối lại)** và đưa qua một **final linear layer (lớp tuyến tính cuối)** để trộn chúng trở lại thành một vector đầy đủ. Phép trộn cuối này cũng được mô hình học.

![Multi-head attention](transformer-multi-head-attention.png "Multi-head attention kết hợp nhiều attention head chuyên biệt.")

Điểm thú vị là các head khác nhau thường trở nên chuyên biệt một phần. Mô hình không được chỉ dẫn mỗi head phải làm gì; sự chuyên biệt tự xuất hiện trong quá trình huấn luyện.

Các nhà nghiên cứu đã phát hiện những head có xu hướng:

- Theo dõi ngữ pháp, chẳng hạn liên kết động từ với tân ngữ hoặc mạo từ với danh từ.
- Xác định đại từ đang đề cập tới tên nào.
- Theo dõi các mẫu vị trí.
- Hoạt động như induction head.
- Thực hiện nhiều vai trò khác.

Một transformer layer có thể có 32 head. Một mô hình hiện đại ở tuyến đầu có hàng chục layer. Vì vậy, một LLM điển hình có tổng cộng hàng nghìn attention head, mỗi head bổ sung một góc nhìn đã học riêng.

Một mối quan tâm thực tế về chi phí đã thúc đẩy thay đổi kiến trúc gần đây. Mỗi head cần giữ Key và Value vector trong bộ nhớ cho tất cả token đã được sinh. Khi một token mới được tạo, mô hình nhờ đó không phải tính lại mọi thứ từ đầu.

Bộ nhớ này được gọi là **KV cache (bộ nhớ đệm Key–Value)** và là chi phí bộ nhớ chính khi chạy LLM với context dài.

> **Giải thích ngắn — KV cache (bộ nhớ đệm Key–Value)**  
> KV cache lưu Key và Value vector cũ trong quá trình sinh. Nó giúp mô hình không phải tính lại toàn bộ prompt mỗi khi thêm một token.

Các LLM decoder-only hiện đại phần lớn sử dụng một biến thể gọi là **Grouped-Query Attention — GQA (attention truy vấn theo nhóm)**. Thay vì mỗi head có Key và Value riêng, nhiều query head được gom nhóm để dùng chung một số lượng key/value head ít hơn.

Ví dụ:

- LLaMA-2 70B có 64 query head nhưng chỉ có 8 key/value head.
- Mistral 7B có 32 query head và 8 key/value head.

Kết quả là độ chính xác gần tương đương full multi-head attention, nhưng áp lực bộ nhớ và chi phí **inference (suy luận)** thấp hơn đáng kể.

> **Giải thích ngắn — GQA**  
> Grouped-Query Attention cho phép nhiều query head dùng chung ít key/value head hơn. Điều này giảm bộ nhớ KV cache trong khi vẫn giữ được nhiều góc nhìn truy vấn.

---

## 6. Feed-forward network (mạng truyền thẳng)

Sau khi attention hoàn tất việc trộn thông tin giữa các token, mỗi layer có một bước thứ hai ít được nhắc đến hơn: **feed-forward network — FFN (mạng truyền thẳng)**.

Nếu attention là quá trình các token “nói chuyện” với nhau, thì FFN là quá trình mỗi token tự thực hiện thêm xử lý. FFN chạy độc lập trên vector của từng token, không trộn thông tin giữa các token.

FFN thực hiện ba bước theo thứ tự:

1. **Expand (mở rộng)** vector token lên kích thước lớn hơn. Transformer gốc sử dụng mức mở rộng 4 lần, trong khi các mô hình SwiGLU hiện đại thường sử dụng tỷ lệ mở rộng khác.
2. Áp dụng một **non-linear function (hàm phi tuyến)**.
3. **Compress (nén)** vector trở lại kích thước ban đầu.

![Feed-forward network](transformer-ffn.png "Feed-forward network mở rộng, biến đổi rồi nén vector của từng token.")

Bước phi tuyến ở giữa thực hiện một vai trò cụ thể đáng để hiểu. **Non-linearity (tính phi tuyến)** là một hàm làm “bẻ cong” đầu vào. Ví dụ đơn giản nhất là **ReLU — Rectified Linear Unit (đơn vị tuyến tính chỉnh lưu)**: nó trả về 0 cho mọi số âm và giữ nguyên các số dương.

> **Giải thích ngắn — non-linearity (tính phi tuyến)**  
> Hàm phi tuyến ngăn mạng bị rút gọn thành một phép biến đổi tuyến tính lớn duy nhất.

Nếu không có bước này, FFN chỉ là hai lớp tuyến tính xếp chồng. Nhưng việc xếp chồng các phép toán tuyến tính thuần túy sẽ bị rút gọn về mặt toán học: hai lớp tuyến tính liên tiếp tương đương một lớp tuyến tính, và một trăm lớp tuyến tính liên tiếp vẫn tương đương một lớp.

Tính phi tuyến ngăn sự rút gọn đó, cho phép FFN thực hiện phép biến đổi phong phú hơn một phép nhân ma trận đơn lẻ.

Transformer gốc sử dụng ReLU. GPT và BERT chuyển sang **GELU — Gaussian Error Linear Unit (đơn vị tuyến tính sai số Gaussian)**. Các mô hình hiện đại như LLaMA, Mistral và PaLM sử dụng **SwiGLU**. Cấu trúc mở rộng rồi nén được giữ nguyên, trong khi hàm phi tuyến là phần tiếp tục được cải tiến.

Phần lớn **parameter (tham số)** trong một **dense transformer model (mô hình transformer dày đặc)** nằm trong FFN, không phải attention. Một phần lớn trọng số của mô hình được đặt trong các feed-forward layer.

Các tham số đó không mang tính chung chung. Đây là nơi lưu giữ phần lớn cấu trúc về sự kiện và ngữ nghĩa mà mô hình đã học. Các nhà nghiên cứu phát hiện một số **neuron (nơ-ron)** trong FFN có liên hệ mạnh với những khái niệm hoặc sự kiện cụ thể.

Ví dụ, một neuron có thể kích hoạt mạnh với văn bản liên quan đến tháp Eiffel, neuron khác với ngôn ngữ lập trình, neuron khác với động từ ở thì quá khứ. Khi mô hình “biết” Paris là thủ đô của Pháp, sự kiện đó được biểu diễn phân tán trong trọng số và **activation (giá trị kích hoạt)** của FFN ở những layer cụ thể.

Tính chất giống bộ nhớ này dẫn tới một hệ quả thú vị: các nhà nghiên cứu đã tìm ra cách chỉnh sửa trực tiếp một số sự kiện trong mô hình đã huấn luyện mà không cần huấn luyện lại toàn bộ mô hình.

Những phương pháp như **ROME — Rank-One Model Editing (chỉnh sửa mô hình hạng một)** có thể thay đổi liên hệ “tháp Eiffel ở Paris” thành “tháp Eiffel ở Rome” bằng một phép chỉnh sửa **low-rank (hạng thấp)** có mục tiêu lên một ma trận trọng số FFN cụ thể. Sau đó, mô hình có xu hướng sinh văn bản phù hợp với liên hệ đã bị chỉnh sửa.

Một số mô hình hiện đại ở tuyến đầu đã bắt đầu thay FFN dày đặc bằng **Mixture of Experts — MoE (hỗn hợp chuyên gia)**.

Thay vì một FFN cho mỗi layer, mô hình có nhiều FFN song song, gọi là **experts (các chuyên gia)**, cùng một **router network (mạng định tuyến)** nhỏ để chọn expert nào sẽ xử lý mỗi token.

Mixtral 8x7B có 8 expert trên mỗi layer nhưng chỉ kích hoạt 2 expert cho một token cụ thể. Tổng số parameter tăng đáng kể, nhưng compute cho mỗi token tăng chậm hơn nhiều vì chỉ một số expert được chạy. Đây là cách tăng quy mô parameter mà không làm chi phí inference tăng theo cùng tỷ lệ.

> **Giải thích ngắn — MoE**  
> Mixture of Experts nghĩa là mô hình có nhiều feed-forward network và chỉ định tuyến mỗi token qua một vài mạng trong số đó.

Mixtral 8x7B có tổng cộng 46,7 tỷ parameter nhưng chỉ sử dụng khoảng 12,9 tỷ parameter cho mỗi token. MoE đã trở thành một lựa chọn phổ biến cho các mô hình rất lớn vì cho phép tiếp tục tăng tổng parameter mà không khiến chi phí inference tăng tỷ lệ thuận.

---

## 7. Residual stream (luồng phần dư) và layer normalization (chuẩn hóa lớp)

**Residual stream (luồng phần dư)** khiến mô hình hoạt động theo kiểu “cộng thêm” thay vì “thay thế”. Sau khi attention hoặc FFN chạy xong, kết quả thường không thay thế vector hiện tại của token. Nó được cộng vào vector đó theo từng vị trí:

```text
vector mới = vector cũ + đầu ra của sub-block
```

> **Giải thích ngắn — residual connection (kết nối phần dư)**  
> Residual connection cộng đầu ra của một block trở lại vector đầu vào của block đó. Nó tạo đường tắt để thông tin và gradient đi xuyên qua mạng.

Qua 30, 50 hoặc 100 layer, đóng góp của từng layer được tích lũy thay vì đơn giản ghi đè lên vector trước đó. Tổng đang chạy này được gọi là residual stream.

Residual stream có một đặc tính đáng chú ý: các input embedding ban đầu vẫn có một đường cộng trực tiếp đi tới những layer rất muộn, đồng thời được trộn với đóng góp của từng sub-block trên đường đi.

![Residual stream](transformer-residual-stream.png "Residual stream tích lũy đầu ra từ attention và feed-forward network.")

Residual connection không được phát minh cho transformer. Nó xuất phát từ **ResNet — Residual Network (mạng phần dư)** của He và cộng sự năm 2015, ban đầu dành cho nhận dạng hình ảnh.

Động lực lúc đó là các mạng sâu gần như không thể huấn luyện. **Training signal (tín hiệu huấn luyện)** trở nên quá yếu — hoặc đôi khi quá mạnh — khi truyền ngược qua nhiều layer. Mô hình không thể học hiệu quả từ sai lầm của chính nó. Việc thêm một đường tắt cho phép tín hiệu đi trực tiếp từ đầu ra trở về đầu vào. Nhờ đó, người ta có thể huấn luyện các mạng có hàng trăm layer. Transformer kế thừa chính thủ thuật này.

Trong nghiên cứu interpretability hiện đại, residual stream trở thành đối tượng trung tâm. Mọi component (thành phần), mọi attention head, mọi FFN và thậm chí cả bước **unembedding (giải nhúng)** ở cuối đều đọc từ residual stream rồi ghi trở lại vào đó.

Thành phần thứ hai, **layer normalization (chuẩn hóa lớp)**, tồn tại vì một lý do thực tế hơn: nếu không có nó, residual stream sẽ không ổn định. Các con số đi qua hàng chục phép cộng có xu hướng tăng bùng nổ hoặc co về gần 0. Cả hai trường hợp đều có thể làm quá trình huấn luyện thất bại.

Layer normalization đưa vector của từng token trở về một phạm vi được kiểm soát giữa các sub-block.

> **Giải thích ngắn — layer normalization (chuẩn hóa lớp)**  
> Layer normalization chuẩn hóa lại vector token để các con số nằm trong phạm vi ổn định trong quá trình huấn luyện.

Transformer gốc năm 2017 áp dụng normalization sau mỗi sub-block, gọi là **post-norm (chuẩn hóa sau)**. Cách này hoạt động với mô hình nông nhưng trở nên khó huấn luyện ổn định khi độ sâu tăng.

Transformer hiện đại — từ GPT-2 trở đi, cùng LLaMA và Mistral — thường áp dụng normalization trước mỗi sub-block, gọi là **pre-norm (chuẩn hóa trước)**. Đây là một trong những thay đổi giúp transformer rất sâu dễ huấn luyện hơn.

Bản thân hàm normalization cũng đã thay đổi. Nhiều mô hình open hiện đại như LLaMA, Mistral, Gemma và Phi sử dụng một biến thể đơn giản hơn gọi là **RMSNorm — Root Mean Square Normalization (chuẩn hóa căn trung bình bình phương)**.

Layer normalization gốc thực hiện đồng thời hai việc:

1. Dịch vector về quanh 0 bằng cách trừ giá trị trung bình.
2. Chuẩn hóa lại độ lớn của các con số.

RMSNorm bỏ bước dịch về 0 và chỉ giữ bước chuẩn hóa độ lớn. Thực nghiệm cho thấy việc chuẩn hóa độ lớn mang lại phần lớn lợi ích nhưng rẻ hơn về compute.

> **Giải thích ngắn — RMSNorm**  
> RMSNorm là phương pháp normalization rẻ hơn, chuẩn hóa độ lớn của vector mà không trừ giá trị trung bình trước.

Đây là phần cơ chế ít hào nhoáng nhưng thiết yếu. Nếu không có residual connection, các mô hình rất sâu khó huấn luyện hơn nhiều. Nếu không có layer normalization, tổng đang tích lũy có thể bùng nổ hoặc co sụp. Khi kết hợp cả hai, ta có thể xây dựng mô hình sâu hàng trăm layer.

---

## 8. Next-token prediction (dự đoán token tiếp theo)

Sau khi tất cả layer attention và feed-forward hoàn tất xử lý, mô hình có một vector cho mỗi token trong chuỗi. Trong quá trình sinh văn bản, để dự đoán token tiếp theo, mô hình chỉ lấy vector cuối cùng của token cuối cùng.

Vector cuối này được chuyển thành một con số cho mỗi token có thể xuất hiện tiếp theo. Nếu vocabulary có 100.000 token, mô hình tạo ra 100.000 con số. Những con số này được gọi là **logits (điểm thô)**. Chúng chưa phải xác suất và có thể mang bất kỳ giá trị dương hoặc âm nào.

> **Giải thích ngắn — logits (điểm thô)**  
> Logits là điểm chưa chuẩn hóa cho từng token có thể xuất hiện tiếp theo. Chúng chỉ trở thành xác suất sau khi đi qua softmax.

Softmax biến logits thành **probability distribution (phân phối xác suất)** của mô hình trên toàn bộ token có thể xuất hiện tiếp theo. Đây là cùng một phép toán softmax đã xuất hiện trước đó, nhưng được sử dụng ở một vị trí khác trong mô hình.

Mô hình thường không luôn chọn token có xác suất cao nhất. Các thiết lập **decoding (giải mã)** kiểm soát mức độ xác định hoặc đa dạng của đầu ra.

- **Temperature (nhiệt độ)** thay đổi độ sắc nét của phân phối.
- **Top-k sampling (lấy mẫu top-k)** giới hạn lựa chọn trong `k` token có xác suất cao nhất.
- **Top-p sampling (lấy mẫu top-p)**, còn gọi là **nucleus sampling (lấy mẫu hạt nhân)**, giữ tập token nhỏ nhất có tổng xác suất đạt ngưỡng `p`.

Đây là lý do cùng một mô hình có thể tạo cảm giác chính xác, thận trọng trong một thiết lập và sáng tạo, đa dạng hơn trong thiết lập khác.

> **Giải thích ngắn — temperature (nhiệt độ)**  
> Temperature kiểm soát độ ngẫu nhiên khi lấy mẫu. Temperature thấp làm mô hình thận trọng và dễ dự đoán hơn; temperature cao làm đầu ra đa dạng hơn.

Sau khi một token được chọn, nó được thêm vào input. Mô hình chạy bước tiếp theo trên chuỗi dài hơn, thường tái sử dụng KV cache để không phải tính lại toàn bộ prefix (tiền tố) từ đầu.

Quá trình lặp lại như sau:

1. Attention mới cho token mới.
2. Feed-forward mới.
3. Vector cuối mới.
4. Dự đoán token mới.

Vòng lặp tiếp tục cho đến khi mô hình sinh **end-of-sequence token — EOS token (token kết thúc chuỗi)** hoặc chạm giới hạn độ dài. Một đoạn văn hoàn chỉnh chỉ là kết quả của vòng lặp này, mỗi lần một token.

Mục tiêu duy nhất — dự đoán token tiếp theo — là tín hiệu huấn luyện cốt lõi của một **base LLM (LLM nền tảng)**. Base model không được huấn luyện trực tiếp để đạt độ chính xác thực tế, hội thoại tốt, reasoning (lập luận) hoặc coding (lập trình). Nó được huấn luyện để dự đoán token tiếp theo trên khối lượng văn bản khổng lồ.

Sau đó, post-training có thể điều chỉnh mô hình để:

- Tuân thủ instruction (chỉ dẫn).
- Phù hợp với preference (sở thích/ưu tiên) của con người.
- Tuân thủ các yêu cầu safety (an toàn).
- Có hành vi hội thoại tốt hơn.

Có một cải tiến lớn về hiệu quả đáng biết là **speculative decoding (giải mã suy đoán)**.

Một mô hình nhỏ và nhanh được dùng làm **draft model (mô hình nháp)** để đề xuất trước nhiều token. Mô hình lớn kiểm tra các token đó song song. Nếu token đề xuất được chấp nhận theo phân phối xác suất của mô hình lớn, chúng được giữ lại. Nếu không, hệ thống quay lại sử dụng token do mô hình lớn tạo ra.

Khi triển khai đúng, phân phối đầu ra vẫn giống như chỉ chạy mô hình lớn, nhưng vòng lặp có thể nhanh hơn đáng kể.

> **Giải thích ngắn — speculative decoding (giải mã suy đoán)**  
> Speculative decoding dùng một mô hình nháp nhỏ để đoán trước, rồi yêu cầu mô hình lớn xác minh nhiều token dự đoán cùng lúc.

Vòng lặp dự đoán token tiếp theo là phần đơn giản nhất của kiến trúc, nhưng chính nó khiến toàn bộ hệ thống hoạt động.

---

## 9. Architecture (kiến trúc) so với trained weights (trọng số đã huấn luyện)

Đến đây, chúng ta đã đi qua các cơ chế cốt lõi:

- Tokens.
- Embeddings.
- Positional encoding.
- Attention.
- Multi-head attention.
- Feed-forward network.
- Residual stream và normalization.
- Vòng lặp next-token prediction ở phía đầu ra.

Đó là kiến trúc cơ bản trong một lượt tổng quan.

Vậy điều gì thực sự khác nhau giữa GPT, Claude, Gemini và LLaMA?

Mức độ công khai chi tiết khác nhau, và các mô hình proprietary (độc quyền) không công bố mọi lựa chọn kiến trúc. Tuy nhiên, ở mức độ được trình bày trong bài này, chúng nhìn chung cùng nằm trong không gian thiết kế của transformer family.

Phần lớn LLM hiện đại dựa trên transformer đều sử dụng cấu trúc tổng quát giống nhau:

1. Tokenization.
2. Embeddings.
3. Positional encoding.
4. Nhiều transformer layer xếp chồng, mỗi layer gồm multi-head attention và feed-forward network.
5. Residual stream.
6. Layer normalization.
7. Next-token prediction.

Những yếu tố thay đổi giữa các mô hình gồm:

- **Trained weights:** các con số được học từ dữ liệu huấn luyện khác nhau, ở những quy mô khác nhau.
- **Configuration (cấu hình):** số layer, vocabulary size, số head, parameter count, sử dụng MoE hay dense architecture.
- **Post-training:** instruction tuning (tinh chỉnh theo chỉ dẫn), học từ human feedback (phản hồi của con người), và các safety control (cơ chế kiểm soát an toàn) áp dụng trên base model.

> **Giải thích ngắn — weights (trọng số)**  
> Weights là những con số được học bên trong mô hình. Quá trình huấn luyện điều chỉnh chúng cho đến khi mô hình có thể dự đoán văn bản tốt.

Trong giai đoạn 2023–2025, “modern transformer stack (ngăn xếp transformer hiện đại)” đã hội tụ về một tập hợp lựa chọn chung trong nhiều mô hình nghiêm túc ở tuyến đầu và nhiều mô hình open-weight, dù các nhóm nghiên cứu đi đến những lựa chọn này theo những con đường khác nhau:

- Pre-norm placement.
- RMSNorm.
- RoPE.
- SwiGLU.
- Grouped-Query Attention.
- Mixture of Experts trong một số mô hình lớn nhất.

Không có kỹ thuật nào trong số này được phát minh cùng lúc. Chúng được tích lũy qua khoảng năm năm cải tiến trên nền thiết kế transformer gốc năm 2017.

---

## 10. Hướng phát triển tiếp theo

Sự hội tụ quanh transformer-family architecture là điều khác thường trong lịch sử machine learning (học máy). Trong phần lớn lịch sử lĩnh vực này, mỗi bài toán có một loại network (mạng) chuyên biệt:

- Image recognition (nhận dạng hình ảnh) sử dụng một loại kiến trúc.
- Language (ngôn ngữ) sử dụng một loại khác.
- Audio (âm thanh) sử dụng loại thứ ba.
- Các nhóm nghiên cứu vision (thị giác máy tính) và language gần như không dùng chung phương pháp.

Ngày nay, các mô hình kiểu transformer xuất hiện trong language, vision, audio và **multimodal system (hệ thống đa phương thức)**. Transformer đã hấp thụ một phần rất lớn của toàn bộ lĩnh vực.

Điều này có thể thay đổi. **Mamba** và các **state-space model — SSM (mô hình không gian trạng thái)** khác là những phương án thay thế đáng tin cậy, đặc biệt với chuỗi rất dài. Các **hybrid architecture (kiến trúc lai)** đang được nghiên cứu. Mixture of Experts cũng đã thay đổi ý nghĩa của “architecture” ở các mô hình tuyến đầu theo những cách từng bị xem là rất khác thường chỉ năm năm trước.

Tuy nhiên, các cơ chế cốt lõi trong bài viết này vẫn là những phần bền vững:

- Tokens.
- Embeddings.
- Positional encoding.
- Attention.
- Feed-forward network.
- Residual stream và normalization.
- Next-token prediction.

Ngay cả khi kiến trúc thay đổi, đây vẫn là những bài toán mà bất kỳ sequence model (mô hình chuỗi) nào cũng phải giải quyết dưới một hình thức nào đó.

Nếu đã đọc đến đây, bạn có thể xem nhiều bài báo transformer hoặc model card hiện đại và biết từng phần đang đề cập đến thành phần nào trong hệ thống. Đó chính là mục tiêu của bài viết.

Phản hồi luôn được hoan nghênh. Nếu bạn quan tâm đến bất kỳ nội dung nào trong bài, hãy liên hệ với tác giả trên X. Tác giả rất thích kết bạn mới.

---

## Tóm tắt transformer pipeline (quy trình transformer)

![Toàn bộ transformer pipeline](transformer-pipeline.png "Toàn bộ transformer pipeline: từ văn bản đầu vào, qua tokenizer, embeddings, các transformer layer, đến logits và token tiếp theo.")
