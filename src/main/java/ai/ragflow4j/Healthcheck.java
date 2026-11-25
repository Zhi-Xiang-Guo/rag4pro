package ai.ragflow4j;

import java.net.URI;
import java.net.http.*;
import java.time.Duration;

/**
 * 容器不需要额外安装 curl，健康检查只验证服务和索引可读，不调用模型。
 */
public final class Healthcheck {
    public static void main(String[] args) {
        try {
            var request =
                    HttpRequest.newBuilder(URI.create("http://127.0.0.1:18080/api/health"))
                            .timeout(Duration.ofSeconds(3))
                            .build();
            int status =
                    HttpClient.newHttpClient()
                            .send(request, HttpResponse.BodyHandlers.discarding())
                            .statusCode();
            System.exit(status == 200 ? 0 : 1);
        } catch (Exception e) {
            System.exit(1);
        }
    }
}
