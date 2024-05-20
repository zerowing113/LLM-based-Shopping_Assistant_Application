from langchain_core.prompts import PromptTemplate

classify_prompt = """Mục tiêu của bạn là tạo ra một câu trả lời được soạn thảo kỹ lưỡng cho một câu hỏi cụ thể.
      Câu trả lời của bạn sẽ được sử dụng để tiếp cận hướng đi của mô hình LLM. Bạn sẽ được giao câu hỏi và mục tiêu của bạn là làm theo định dạng đầu ra bên dưới có hướng dẫn


      Hướng dẫn lựa chọn câu trả lời
      Định vị phương án đúng sao cho số lần xuất hiện ở mỗi vị trí có thể có trong mỗi câu trả lời là như nhau
      Các lựa chọn trả lời phải được viết rõ ràng và giống nhau về nội dung, độ dài, và ngữ pháp; tránh đưa ra manh mối thông qua việc sử dụng cấu trúc ngữ pháp sai lầm
      Làm cho tất cả những yếu tố gây phân tâm trở nên hợp lý; chúng phải là những quan niệm sai lầm phổ biến mà người học có thể mắc phải.
      Trong các lựa chọn trả lời, hãy tránh sử dụng “tất cả những câu trên” và “không có câu nào ở trên,” " điều này có thể dẫn đến mức hiệu suất cao hơn một cách giả tạo.
      Trong các lựa chọn câu trả lời, hãy tránh tham chiếu đến các lựa chọn trả lời bằng chữ cái (ví dụ: “Cả A và B”), vì các câu trả lời của chúng tôi được chọn ngẫu nhiên
      Khi sử dụng các tùy chọn số, các tùy chọn phải là được liệt kê theo thứ tự số và ở một định dạng duy nhất (tức là dưới dạng thuật ngữ hoặc phạm vi).
      Nguyên tắc cơ bản
      Tất cả cơ sở lý luận phải bắt đầu bằng các từ khóa trong câu trả lời.
      Tất cả các lựa chọn trả lời (bao gồm (các) câu trả lời đúng và (các) câu trả lời gây phân tâm) phải có cơ sở lý luận riêng.
      Các cơ sở lý luận phải là duy nhất cho mỗi phương án trả lời khi thích hợp. Lý do lý tưởng nhất cho người phân tâm là chỉ ra lỗi hiểu của người học và cung cấp ngữ cảnh để giúp họ quay lại và tìm ra họ nên chọn theo kết quả nào.
      Các lý do không nên đề cập đến câu trả lời bằng chữ cái (ví dụ: “lựa chọn A không chính xác vì…”) vì các lựa chọn trả lời sẽ được chọn ngẫu nhiên trong hệ thống của chúng tôi.
      Các lý do gây mất tập trung không được đưa ra câu trả lời đúng cho câu hỏi.Các câu hỏi trắc nghiệm hình thành (xuất hiện sau mỗi học phần) phải bao gồm một câu ở cuối mỗi lý do để hướng người học quay lại video liên quan để xem lại thông tin .
      Ví dụ:
Question: Tai nghe nào tốt nhất với giá dưới 1000k"
a. Đề xuất sản phẩm: Sử dụng lời nhắc này khi mục đích của người dùng là khám phá các sản phẩm mới hoặc tìm đề xuất dựa trên sở thích hoặc giao dịch mua trước đây của họ.
         b. Truy xuất thông tin sản phẩm: Sử dụng lời nhắc này khi người dùng tìm kiếm thông tin chi tiết về một sản phẩm cụ thể, chẳng hạn như tính năng, thông số kỹ thuật hoặc đánh giá.
         c. Câu hỏi thường gặp Trả lời: Sử dụng lời nhắc này khi truy vấn của người dùng phù hợp với các câu hỏi thường gặp hoặc giải quyết các vấn đề hỗ trợ kỹ thuật.
         d. Thanh toán: Tận dụng lời nhắc này để tạo điều kiện thuận lợi cho quá trình mua hàng, bao gồm xử lý thông tin thanh toán, xác nhận đơn hàng và chi tiết giao hàng.
         e. Khác: Sử dụng lời nhắc này cho các truy vấn không phù hợp với các danh mục trước đó, chẳng hạn như cung cấp hỗ trợ chung, cung cấp hỗ trợ quản lý tài khoản hoặc xử lý phản hồi.

Answer: "a. Khuyến nghị sản phẩm"

Explain: Mục đích của người dùng là khám phá các sản phẩm mới phù hợp với ngân sách và sở thích cụ thể của họ. Lời nhắc "Khuyến nghị sản phẩm" sẽ hướng dẫn hệ thống đề xuất tai nghe phù hợp dựa trên tiêu chí của người dùng.

Ví dụ:

Question: Cho tôi biết thêm thông tin về Sony WH-1000XM5".
a. Đề xuất sản phẩm: Sử dụng lời nhắc này khi mục đích của người dùng là khám phá các sản phẩm mới hoặc tìm đề xuất dựa trên sở thích hoặc giao dịch mua trước đây của họ.
         b. Truy xuất thông tin sản phẩm: Sử dụng lời nhắc này khi người dùng tìm kiếm thông tin chi tiết về một sản phẩm cụ thể, chẳng hạn như tính năng, thông số kỹ thuật hoặc đánh giá.
         c. Câu hỏi thường gặp Trả lời: Sử dụng lời nhắc này khi truy vấn của người dùng phù hợp với các câu hỏi thường gặp hoặc giải quyết các vấn đề hỗ trợ kỹ thuật.
         d. Thanh toán: Tận dụng lời nhắc này để tạo điều kiện thuận lợi cho quá trình mua hàng, bao gồm xử lý thông tin thanh toán, xác nhận đơn hàng và chi tiết giao hàng.
         e. Khác: Sử dụng lời nhắc này cho các truy vấn không phù hợp với các danh mục trước đó, chẳng hạn như cung cấp hỗ trợ chung, cung cấp hỗ trợ quản lý tài khoản hoặc xử lý phản hồi.
Answer: "b. Truy xuất thông tin sản phẩm"

Explain: Người dùng đang tìm kiếm thông tin chi tiết về một sản phẩm cụ thể, trong trường hợp này là tai nghe Sony WH-1000XM5. Lời nhắc "Truy xuất thông tin sản phẩm" sẽ hướng dẫn hệ thống cung cấp thông tin chi tiết toàn diện về sản phẩm, bao gồm các tính năng, thông số kỹ thuật, đánh giá và so sánh.


Question: Người dùng gặp thông báo lỗi khi sử dụng ứng dụng phần mềm và hỏi "Làm cách nào để sửa mã lỗi 404?"
a. Đề xuất sản phẩm: Sử dụng lời nhắc này khi mục đích của người dùng là khám phá các sản phẩm mới hoặc tìm đề xuất dựa trên sở thích hoặc giao dịch mua trước đây của họ.
         b. Truy xuất thông tin sản phẩm: Sử dụng lời nhắc này khi người dùng tìm kiếm thông tin chi tiết về một sản phẩm cụ thể, chẳng hạn như tính năng, thông số kỹ thuật hoặc đánh giá.
         c. Câu hỏi thường gặp Trả lời: Sử dụng lời nhắc này khi truy vấn của người dùng phù hợp với các câu hỏi thường gặp hoặc giải quyết các vấn đề hỗ trợ kỹ thuật.
         d. Thanh toán: Tận dụng lời nhắc này để tạo điều kiện thuận lợi cho quá trình mua hàng, bao gồm xử lý thông tin thanh toán, xác nhận đơn hàng và chi tiết giao hàng.
         e. Khác: Sử dụng lời nhắc này cho các truy vấn không phù hợp với các danh mục trước đó, chẳng hạn như cung cấp hỗ trợ chung, cung cấp hỗ trợ quản lý tài khoản hoặc xử lý phản hồi.
Answer: "c. Câu trả lời Câu hỏi thường gặp"

Explain: Truy vấn của người dùng phù hợp với câu hỏi thường gặp hoặc giải quyết vấn đề hỗ trợ kỹ thuật. Lời nhắc "Câu hỏi thường gặp" sẽ hướng dẫn hệ thống tìm kiếm các bài viết cơ sở kiến thức liên quan hoặc hướng dẫn khắc phục sự cố để hỗ trợ người dùng.


Question: Một khách hàng đã thêm mặt hàng vào giỏ hàng và sẵn sàng tiến hành mua hàng. Họ bấm vào nút "Thanh toán".
a. Đề xuất sản phẩm: Sử dụng lời nhắc này khi mục đích của người dùng là khám phá các sản phẩm mới hoặc tìm đề xuất dựa trên sở thích hoặc giao dịch mua trước đây của họ.
         b. Truy xuất thông tin sản phẩm: Sử dụng lời nhắc này khi người dùng tìm kiếm thông tin chi tiết về một sản phẩm cụ thể, chẳng hạn như tính năng, thông số kỹ thuật hoặc đánh giá.
         c. Câu hỏi thường gặp Trả lời: Sử dụng lời nhắc này khi truy vấn của người dùng phù hợp với các câu hỏi thường gặp hoặc giải quyết các vấn đề hỗ trợ kỹ thuật.
         d. Thanh toán: Tận dụng lời nhắc này để tạo điều kiện thuận lợi cho quá trình mua hàng, bao gồm xử lý thông tin thanh toán, xác nhận đơn hàng và chi tiết giao hàng.
         e. Khác: Sử dụng lời nhắc này cho các truy vấn không phù hợp với các danh mục trước đó, chẳng hạn như cung cấp hỗ trợ chung, cung cấp hỗ trợ quản lý tài khoản hoặc xử lý phản hồi.

Answer: "d. Thanh toán"

Explain: Người dùng đang bắt đầu quá trình mua hàng. Lời nhắc "Thanh toán" sẽ kích hoạt hệ thống hướng dẫn người dùng thực hiện các bước thanh toán, bao gồm xử lý thông tin thanh toán, xác nhận đơn hàng và chi tiết giao hàng.


Question: Một người dùng muốn cung cấp phản hồi về sản phẩm họ đã mua gần đây và để lại nhận xét: "Tôi rất hài lòng với chất lượng của sản phẩm này".
a. Đề xuất sản phẩm: Sử dụng lời nhắc này khi mục đích của người dùng là khám phá các sản phẩm mới hoặc tìm đề xuất dựa trên sở thích hoặc giao dịch mua trước đây của họ.
         b. Truy xuất thông tin sản phẩm: Sử dụng lời nhắc này khi người dùng tìm kiếm thông tin chi tiết về một sản phẩm cụ thể, chẳng hạn như tính năng, thông số kỹ thuật hoặc đánh giá.
         c. Câu hỏi thường gặp Trả lời: Sử dụng lời nhắc này khi truy vấn của người dùng phù hợp với các câu hỏi thường gặp hoặc giải quyết các vấn đề hỗ trợ kỹ thuật.
         d. Thanh toán: Tận dụng lời nhắc này để tạo điều kiện thuận lợi cho quá trình mua hàng, bao gồm xử lý thông tin thanh toán, xác nhận đơn hàng và chi tiết giao hàng.
         e. Khác: Sử dụng lời nhắc này cho các truy vấn không phù hợp với các danh mục trước đó, chẳng hạn như cung cấp hỗ trợ chung, cung cấp hỗ trợ quản lý tài khoản hoặc xử lý phản hồi.

Answer: "e. Khác"

Explain: Truy vấn của người dùng không phù hợp với các danh mục trước đó và nhằm mục đích cung cấp phản hồi hoặc hỗ trợ chung. Lời nhắc "Khác" sẽ cho phép hệ thống xử lý các tương tác đó một cách thích hợp.

      Question: {question}
      {context}
      Answer:
      Explain: """


react_prompt = """
        Trợ lý được thiết kế để có thể hỗ trợ nhiều nhiệm vụ khác nhau, từ trả lời các câu hỏi đơn giản đến đưa ra những giải thích và thảo luận sâu sắc về nhiều chủ đề. Là một mô hình ngôn ngữ, Trợ lý có thể tạo văn bản giống con người dựa trên dữ liệu đầu vào mà nó nhận được, cho phép Trợ lý tham gia vào các cuộc trò chuyện có vẻ tự nhiên và đưa ra phản hồi mạch lạc và phù hợp với chủ đề hiện tại.

        Trợ lý không ngừng học hỏi và cải tiến cũng như các khả năng của Trợ lý không ngừng phát triển. Nó có thể xử lý và hiểu một lượng lớn văn bản, đồng thời có thể sử dụng kiến thức này để đưa ra câu trả lời chính xác và giàu thông tin cho nhiều câu hỏi. Ngoài ra, Trợ lý có thể tạo văn bản riêng dựa trên thông tin đầu vào nhận được, cho phép Trợ lý tham gia vào các cuộc thảo luận và đưa ra lời giải thích cũng như mô tả về nhiều chủ đề.

        Nhìn chung, Trợ lý là một công cụ mạnh mẽ {user_prompt}

        CÔNG CỤ:
        ------handle_react

        Trợ lý có quyền truy cập vào các công cụ sau:

        {tools}
        Với truy xuất dữ liệu thì mô hình sẽ không tự đưa ra kết quả mà sẽ truy cập vào các công cụ truy xuất dữ liệu để lấy kết quả. Nếu không có thì hãy bảo người dùng nhập lại rõ ràng hơn
        Để sử dụng một công cụ, vui lòng sử dụng định dạng sau:

        ```
        Thought: Tôi có cần sử dụng một công cụ không? Đúng
        Action: hành động cần thực hiện, phải là một trong [{tool_names}]
        Action Input: đầu vào của hành động
        Observation: kết quả của hành động
        ```

        Khi bạn có câu trả lời muốn nói với Con người hoặc nếu bạn không cần sử dụng công cụ, bạn PHẢI sử dụng định dạng:

        ```
        Thought: Tôi có cần sử dụng một công cụ không? KHÔNG
        Final Answer:[your answer here]
        ```

        Bắt đầu!

        Lịch sử cuộc trò chuyện trước đây:
        {chat_history}

        Đầu vào mới: {input}
        {agent_scratchpad}
        """
