FROM golang:1.27.3-trixie AS jsonrepair
WORKDIR /src
COPY native/jsonrepair/ ./
RUN CGO_ENABLED=0 go build -trimpath -o /ragflow-jsonrepair .

FROM maven:3.9-eclipse-temurin-21 AS java-base
FROM java-base AS build
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends g++ libpcre2-dev libsimde-dev && rm -rf /var/lib/apt/lists/*
COPY native/upstream native/upstream
COPY native/free_string.cpp native/free_string.cpp
COPY scripts/native-build.sh scripts/native-build.sh
RUN bash scripts/native-build.sh
COPY --from=jsonrepair /ragflow-jsonrepair /build/native/build/ragflow-jsonrepair
COPY native/resources native/resources
COPY pom.xml ./
COPY src ./src
RUN --mount=type=cache,target=/root/.m2 MAVEN_OPTS="-Xmx256m" mvn -B -q -DargLine=-Xmx768m package && java -Xmx768m -jar target/evidence-rag-java.jar --check-runtime

# The C++ analyzer is linked against the build image's glibc and libstdc++.
FROM java-base AS runtime
WORKDIR /app
RUN groupadd --gid 10001 rag && useradd --uid 10001 --gid rag --no-create-home rag \
    && mkdir -p /app/data && chown -R rag:rag /app
COPY --from=build --chown=rag:rag /build/target/evidence-rag-java.jar /app/evidence-rag-java.jar
COPY --from=build --chown=rag:rag /build/native/build/libragflow_tokenizer.so /app/native/build/libragflow_tokenizer.so
COPY --from=jsonrepair --chown=rag:rag /ragflow-jsonrepair /app/native/build/ragflow-jsonrepair
COPY --from=build --chown=rag:rag /build/native/resources /app/native/resources
USER rag
RUN java -Xmx768m -jar /app/evidence-rag-java.jar --check-runtime
ENV EVIDENCE_RAG_HOST=0.0.0.0 EVIDENCE_RAG_PORT=18080 EVIDENCE_RAG_DATA_DIR=/app/data
EXPOSE 18080
ENTRYPOINT ["java", "-XX:MaxRAMPercentage=70.0", "-jar", "/app/evidence-rag-java.jar"]
